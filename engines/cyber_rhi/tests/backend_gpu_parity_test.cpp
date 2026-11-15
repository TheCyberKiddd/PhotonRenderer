// The same RHI frame, recorded once and run on every compiled-in hardware backend:
// bind groups, pipeline layouts, compute and graphics pipelines, command lists,
// barriers, copies, submission, fences and swapchains. Phase 34's done line is "the
// same renderer code on D3D12 and Vulkan"; this is that sentence as a test. Only the
// shader bytes differ, because the project's compiler has no SPIR-V output yet: the
// D3D12 backend compiles HLSL, the Vulkan one is given SPIR-V written with
// `SpirvBuilder`. Everything else, and every assertion, is shared.

#include "spirv_test_shaders.h"

#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/platform/platform.h>
#include <cyber/rhi/backend.h>
#include <cyber/rhi/d3d12.h>
#include <cyber/rhi/device.h>
#include <cyber/rhi/rhi.h>
#include <cyber/rhi/vulkan.h>
#include <cyber/test/test.h>

#include <cstdio>
#include <cstring>
#include <utility>

using namespace cyber::rhi;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;
using cyber::memory::Byte;
using cyber::memory::DynamicArray;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;

namespace
{

constexpr u32 kTarget = 64;
constexpr u32 kPixelBytes = kTarget * kTarget * 4;
constexpr u32 kElements = 1000;

// HLSL twins of the SPIR-V test shaders.
constexpr const char* kComputeHlsl = R"(
StructuredBuffer<uint> inb : register(t0, space0);
RWStructuredBuffer<uint> outb : register(u0, space0);
cbuffer PC : register(b0, space0) { uint count; uint scale; uint bias; };
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x < count) outb[id.x] = inb[id.x] * scale + bias;
}
)";

constexpr const char* kVertexHlsl = R"(
struct VSOut { float4 position : SV_Position; float3 color : COLOR0; };
VSOut main(uint id : SV_VertexID)
{
    VSOut o;
    o.position = float4(id == 1 ? 3.0 : -1.0, id == 2 ? 3.0 : -1.0, 0.0, 1.0);
    o.color = float3(id == 0 ? 1.0 : 0.0, id == 1 ? 1.0 : 0.0, id == 2 ? 1.0 : 0.0);
    return o;
}
)";

constexpr const char* kPixelHlsl = R"(
float4 main(float4 position : SV_Position, float3 color : COLOR0) : SV_Target
{
    return float4(color, 1.0);
}
)";

struct Gpu
{
    HeapAllocator heap;
    Device device;
    BackendKind kind = BackendKind::Null;
    // One fence for the whole test, kept alive until the end: the RHI proves a submission
    // finished by the fence it signalled, so destroying the fence early leaves every
    // resource that submission used "in flight".
    FenceHandle fence{};
    u64 fenceValue = 0;
};

[[nodiscard]] bool Open(const Backend* backend, Gpu& gpu) noexcept
{
    if (backend == nullptr || backend->enumerateAdapters(nullptr, 0) == 0)
    {
        std::printf("  skipped: no adapter\n");
        return false;
    }
    DeviceDesc desc{};
    desc.backend = backend;
    desc.validation = ValidationLevel::Standard;
    auto created = Device::Create(desc, gpu.heap);
    if (!created)
    {
        std::printf("  skipped: %s\n", created.Error().message);
        return false;
    }
    gpu.device = std::move(created.Value());
    gpu.kind = backend->kind;
    return true;
}

// A shader for the backend under test: HLSL source for D3D12, SPIR-V for Vulkan.
[[nodiscard]] ShaderHandle MakeShader(Gpu& gpu, ShaderStage stage, const char* hlsl,
                                      const DynamicArray<u32>& spirv, const char* label) noexcept
{
    ShaderDesc desc{};
    desc.stage = stage;
    desc.entryPoint = "main";
    desc.label = label;
    if (gpu.kind == BackendKind::Vulkan)
    {
        desc.format = ShaderBytecodeFormat::SpirV;
        desc.code = Span<const Byte>(reinterpret_cast<const Byte*>(spirv.Data()), spirv.Size() * 4);
    }
    else
    {
        desc.format = ShaderBytecodeFormat::HlslSource;
        desc.code = Span<const Byte>(reinterpret_cast<const Byte*>(hlsl), std::strlen(hlsl));
    }
    auto created = gpu.device.CreateShader(desc);
    if (!created)
    {
        std::printf("  shader '%s' failed: %s\n", label, created.Error().message);
        return ShaderHandle{};
    }
    return created.Value();
}

// Records the first failure of a recording step, with the backend's own message, so a
// failing backend says which call it refused.
void Check(bool& ok, const char* what, const cyber::core::Status<cyber::core::Error>& status) noexcept
{
    if (ok && !status)
    {
        std::printf("  %s failed: %s\n", what, status.Error().message);
        ok = false;
    }
}

[[nodiscard]] bool SubmitAndWait(Gpu& gpu, CommandList& list) noexcept
{
    if (!gpu.fence.IsValid())
    {
        auto fence = gpu.device.CreateFence(FenceDesc{});
        if (!fence)
        {
            return false;
        }
        gpu.fence = fence.Value();
    }
    const u64 value = ++gpu.fenceValue;
    const CommandListHandle lists[1] = {list.Handle()};
    const FenceSignal signals[1] = {{gpu.fence, value}};
    SubmitDesc submit{};
    submit.queue = QueueKind::Graphics;
    submit.commandLists = Span<const CommandListHandle>(lists, 1);
    submit.signals = Span<const FenceSignal>(signals, 1);
    return gpu.device.Submit(submit).IsOk() &&
           gpu.device.WaitFence(gpu.fence, value, 10'000'000'000ULL).IsOk();
}

void ReleaseFence(Gpu& gpu) noexcept
{
    if (gpu.fence.IsValid())
    {
        (void)gpu.device.DestroyFence(gpu.fence);
        gpu.fence = FenceHandle{};
    }
}

[[nodiscard]] BufferHandle MakeBuffer(Gpu& gpu, BufferUsage usage, u64 bytes, u32 elementStride = 0) noexcept
{
    BufferDesc desc{};
    desc.usage = usage;
    desc.sizeBytes = bytes;
    desc.elementStride = elementStride;
    auto created = gpu.device.CreateBuffer(desc);
    return created ? created.Value() : BufferHandle{};
}

// ---- Compute: bind groups, push constants, dispatch, buffer copies ---------

[[nodiscard]] bool RunCompute(Gpu& gpu, u32 (&result)[kElements]) noexcept
{
    Device& device = gpu.device;
    DynamicArray<u32> spirv(gpu.heap);
    {
        cyber::rhi::detail::SpirvBuilder builder(gpu.heap);
        if (!cyber::rhi::test::BuildScaleBiasCompute(builder, spirv))
        {
            return false;
        }
    }
    const ShaderHandle shader =
        MakeShader(gpu, ShaderStage::Compute, kComputeHlsl, spirv, "scale-bias");
    if (!shader.IsValid())
    {
        return false;
    }

    const BindGroupLayoutEntry entries[2] = {
        {0, BindingType::ReadOnlyStorageBuffer, ShaderStageBit(ShaderStage::Compute), 1,
         kNoShaderRegister, false},
        {1, BindingType::StorageBuffer, ShaderStageBit(ShaderStage::Compute), 1, kNoShaderRegister,
         false},
    };
    BindGroupLayoutDesc layoutDesc{};
    layoutDesc.entries = Span<const BindGroupLayoutEntry>(entries, 2);
    layoutDesc.label = "compute layout";
    auto groupLayout = device.CreateBindGroupLayout(layoutDesc);
    if (!groupLayout)
    {
        std::printf("  bind group layout failed: %s\n", groupLayout.Error().message);
        return false;
    }

    PipelineLayoutDesc pipelineLayoutDesc{};
    pipelineLayoutDesc.groups[0] = groupLayout.Value();
    const PushConstantRange range{0, 12, ShaderStageBit(ShaderStage::Compute)};
    pipelineLayoutDesc.pushConstants = Span<const PushConstantRange>(&range, 1);
    auto pipelineLayout = device.CreatePipelineLayout(pipelineLayoutDesc);
    if (!pipelineLayout)
    {
        std::printf("  pipeline layout failed: %s\n", pipelineLayout.Error().message);
        return false;
    }
    ComputePipelineDesc pipelineDesc{};
    pipelineDesc.layout = pipelineLayout.Value();
    pipelineDesc.computeShader = shader;
    pipelineDesc.label = "scale-bias pipeline";
    auto pipeline = device.CreateComputePipeline(pipelineDesc);
    if (!pipeline)
    {
        std::printf("  compute pipeline failed: %s\n", pipeline.Error().message);
        return false;
    }

    const u64 bytes = kElements * 4ULL;
    const BufferHandle upload =
        MakeBuffer(gpu, BufferUsage::HostVisible | BufferUsage::CopySource, bytes);
    const BufferHandle input =
        MakeBuffer(gpu, BufferUsage::Storage | BufferUsage::CopyDestination, bytes, 4);
    const BufferHandle output =
        MakeBuffer(gpu, BufferUsage::Storage | BufferUsage::CopySource, bytes, 4);
    const BufferHandle readback =
        MakeBuffer(gpu, BufferUsage::Readback | BufferUsage::CopyDestination, bytes);
    if (!upload.IsValid() || !input.IsValid() || !output.IsValid() || !readback.IsValid())
    {
        return false;
    }
    u32 source[kElements];
    for (u32 index = 0; index < kElements; ++index)
    {
        source[index] = index * 3U + 1U;
    }
    if (!device.WriteBuffer(upload, 0, Span<const Byte>(reinterpret_cast<const Byte*>(source), bytes))
             .IsOk())
    {
        return false;
    }

    BindingResource resources[2] = {};
    resources[0].type = BindingType::ReadOnlyStorageBuffer;
    resources[0].buffer = input;
    resources[0].size = bytes;
    resources[1].type = BindingType::StorageBuffer;
    resources[1].buffer = output;
    resources[1].size = bytes;
    BindGroupDesc groupDesc{};
    groupDesc.layout = groupLayout.Value();
    groupDesc.resources = Span<const BindingResource>(resources, 2);
    auto group = device.CreateBindGroup(groupDesc);
    if (!group)
    {
        std::printf("  bind group failed: %s\n", group.Error().message);
        return false;
    }

    CommandListDesc listDesc{};
    listDesc.label = "compute list";
    auto created = device.CreateCommandList(listDesc);
    if (!created)
    {
        std::printf("  command list failed: %s\n", created.Error().message);
        return false;
    }
    CommandList list = std::move(created.Value());
    bool ok = true;
    Check(ok, "Begin", list.Begin());

    // The RHI wants a copy source in exactly `CopySource`, so the upload buffer is moved
    // there from whatever state the backend created it in.
    const BufferBarrier before[3] = {
        {upload, device.BufferStateOf(upload), ResourceState::CopySource},
        {input, ResourceState::Undefined, ResourceState::CopyDestination},
        {output, ResourceState::Undefined, ResourceState::ShaderWrite}};
    Check(ok, "Barrier", list.Barrier(Span<const BufferBarrier>(before, 3)));
    Check(ok, "CopyBuffer", list.CopyBuffer(BufferCopy{upload, 0, input, 0, bytes}));
    const BufferBarrier toRead[1] = {{input, ResourceState::CopyDestination, ResourceState::ShaderRead}};
    Check(ok, "Barrier", list.Barrier(Span<const BufferBarrier>(toRead, 1)));
    Check(ok, "SetPipeline", list.SetPipeline(pipeline.Value()));
    Check(ok, "SetBindGroup", list.SetBindGroup(BindGroupSlot::PerFrame, group.Value()));
    const u32 constants[3] = {kElements, 3, 1};
    // Same data as `source`: out = in * 3 + 1 where in = i * 3 + 1.
    Check(ok, "PushConstants", list.PushConstants(ShaderStageBit(ShaderStage::Compute), 0,
                                  Span<const Byte>(reinterpret_cast<const Byte*>(constants), 12)));
    Check(ok, "Dispatch", list.Dispatch(DispatchArguments{(kElements + 63) / 64, 1, 1}));
    const BufferBarrier toCopy[1] = {{output, ResourceState::ShaderWrite, ResourceState::CopySource}};
    Check(ok, "Barrier", list.Barrier(Span<const BufferBarrier>(toCopy, 1)));
    Check(ok, "CopyBuffer", list.CopyBuffer(BufferCopy{output, 0, readback, 0, bytes}));
    Check(ok, "End", list.End());
    ok = ok && SubmitAndWait(gpu, list);
    ok = ok && device
                   .ReadBuffer(readback, 0,
                               Span<Byte>(reinterpret_cast<Byte*>(result), bytes))
                   .IsOk();

    (void)device.DestroyBindGroup(group.Value());
    {
        const auto destroyed = device.DestroyBuffer(upload);
        if (!destroyed)
        {
            std::printf("  destroy upload: %s\n", destroyed.Error().message);
        }
    }
    (void)device.DestroyBuffer(input);
    (void)device.DestroyBuffer(output);
    (void)device.DestroyBuffer(readback);
    (void)device.DestroyComputePipeline(pipeline.Value());
    (void)device.DestroyPipelineLayout(pipelineLayout.Value());
    (void)device.DestroyBindGroupLayout(groupLayout.Value());
    (void)device.DestroyShader(shader);
    return ok;
}

// ---- Graphics: render pass, pipeline state, draw, texture readback ----------

[[nodiscard]] bool RunGraphics(Gpu& gpu, u8 (&pixels)[kPixelBytes]) noexcept
{
    Device& device = gpu.device;
    DynamicArray<u32> vertexCode(gpu.heap);
    DynamicArray<u32> pixelCode(gpu.heap);
    {
        cyber::rhi::detail::SpirvBuilder vertexBuilder(gpu.heap);
        cyber::rhi::detail::SpirvBuilder pixelBuilder(gpu.heap);
        if (!cyber::rhi::test::BuildTriangleVertex(vertexBuilder, vertexCode) ||
            !cyber::rhi::test::BuildPlainColorFragment(pixelBuilder, pixelCode))
        {
            return false;
        }
    }
    const ShaderHandle vertex =
        MakeShader(gpu, ShaderStage::Vertex, kVertexHlsl, vertexCode, "triangle vs");
    const ShaderHandle pixel =
        MakeShader(gpu, ShaderStage::Pixel, kPixelHlsl, pixelCode, "triangle ps");
    if (!vertex.IsValid() || !pixel.IsValid())
    {
        return false;
    }
    PipelineLayoutDesc layoutDesc{};
    auto layout = device.CreatePipelineLayout(layoutDesc);
    if (!layout)
    {
        std::printf("  empty pipeline layout failed: %s\n", layout.Error().message);
        return false;
    }
    GraphicsPipelineDesc pipelineDesc{};
    pipelineDesc.layout = layout.Value();
    pipelineDesc.vertexShader = vertex;
    pipelineDesc.pixelShader = pixel;
    // A pipeline declares at least one vertex binding even when the shader reads
    // only `SV_VertexID`.
    const VertexBinding vertexBinding{16, VertexInputRate::PerVertex};
    pipelineDesc.vertexLayout.bindings = Span<const VertexBinding>(&vertexBinding, 1);
    const VertexAttribute vertexAttribute{0, 0, Format::R32Float, 0};
    pipelineDesc.vertexLayout.attributes = Span<const VertexAttribute>(&vertexAttribute, 1);
    pipelineDesc.rasterizer.cullMode = CullMode::None;
    pipelineDesc.depthStencil.depthTestEnabled = false;
    pipelineDesc.depthStencil.depthWriteEnabled = false;
    const ColorAttachmentState attachment{Format::RGBA8Unorm, BlendState{}};
    pipelineDesc.colorAttachments = Span<const ColorAttachmentState>(&attachment, 1);
    pipelineDesc.label = "triangle pipeline";
    auto pipeline = device.CreateGraphicsPipeline(pipelineDesc);
    if (!pipeline)
    {
        std::printf("  graphics pipeline failed: %s\n", pipeline.Error().message);
        return false;
    }

    TextureDesc targetDesc{};
    targetDesc.format = Format::RGBA8Unorm;
    targetDesc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
    targetDesc.width = kTarget;
    targetDesc.height = kTarget;
    auto target = device.CreateTexture(targetDesc);
    if (!target)
    {
        return false;
    }
    TextureViewDesc viewDesc{};
    viewDesc.texture = target.Value();
    viewDesc.format = Format::RGBA8Unorm;
    viewDesc.range.mipLevelCount = 1;
    viewDesc.range.layerCount = 1;
    auto view = device.CreateTextureView(viewDesc);
    const BufferHandle readback =
        MakeBuffer(gpu, BufferUsage::Readback | BufferUsage::CopyDestination, kPixelBytes);
    // The pipeline declares one vertex binding, so a buffer is bound to it even though
    // the shader generates its own positions.
    const BufferHandle vertexBuffer = MakeBuffer(gpu, BufferUsage::Vertex, 64);
    if (!view || !readback.IsValid() || !vertexBuffer.IsValid())
    {
        return false;
    }

    CommandListDesc listDesc{};
    listDesc.label = "graphics list";
    auto created = device.CreateCommandList(listDesc);
    if (!created)
    {
        return false;
    }
    CommandList list = std::move(created.Value());
    SubresourceRange whole{};
    whole.mipLevelCount = 1;
    whole.layerCount = 1;
    bool ok = true;
    Check(ok, "Begin", list.Begin());
    const TextureBarrier toTarget[1] = {
        {target.Value(), whole, ResourceState::Undefined, ResourceState::RenderTarget}};
    const BufferBarrier toVertex[1] = {
        {vertexBuffer, ResourceState::Undefined, ResourceState::VertexBuffer}};
    Check(ok, "Barrier",
          list.Barrier(Span<const BufferBarrier>(toVertex, 1), Span<const TextureBarrier>(toTarget, 1)));

    ColorAttachment color{};
    color.view = view.Value();
    color.loadOp = LoadOp::Clear;
    color.storeOp = StoreOp::Store;
    color.clearValue = ClearColorValue{0.0F, 0.0F, 0.0F, 1.0F};
    RenderPassDesc pass{};
    pass.colorAttachments = Span<const ColorAttachment>(&color, 1);
    pass.label = "triangle pass";
    Check(ok, "BeginRenderPass", list.BeginRenderPass(pass));
    Check(ok, "SetPipeline", list.SetPipeline(pipeline.Value()));
    Check(ok, "SetVertexBuffer", list.SetVertexBuffer(0, vertexBuffer, 0));
    Check(ok, "SetViewport", list.SetViewport(Viewport{0, 0, static_cast<float>(kTarget),
                                         static_cast<float>(kTarget), 0.0F, 1.0F}));
    Check(ok, "SetScissor", list.SetScissor(ScissorRect{0, 0, kTarget, kTarget}));
    Check(ok, "Draw", list.Draw(DrawArguments{3, 1, 0, 0}));
    Check(ok, "EndRenderPass", list.EndRenderPass());

    const TextureBarrier toCopy[1] = {
        {target.Value(), whole, ResourceState::RenderTarget, ResourceState::CopySource}};
    Check(ok, "Barrier", list.Barrier({}, Span<const TextureBarrier>(toCopy, 1)));
    BufferTextureCopy copy{};
    copy.buffer = readback;
    copy.bufferRowPitch = kTarget * 4;
    copy.bufferDepthPitch = kTarget * 4 * kTarget;
    copy.texture = target.Value();
    copy.width = kTarget;
    copy.height = kTarget;
    Check(ok, "CopyTextureToBuffer", list.CopyTextureToBuffer(copy));
    Check(ok, "End", list.End());
    ok = ok && SubmitAndWait(gpu, list);
    ok = ok && device
                   .ReadBuffer(readback, 0, Span<Byte>(reinterpret_cast<Byte*>(pixels), kPixelBytes))
                   .IsOk();

    (void)device.DestroyBuffer(vertexBuffer);
    (void)device.DestroyBuffer(readback);
    (void)device.DestroyTextureView(view.Value());
    (void)device.DestroyTexture(target.Value());
    (void)device.DestroyGraphicsPipeline(pipeline.Value());
    (void)device.DestroyPipelineLayout(layout.Value());
    (void)device.DestroyShader(vertex);
    (void)device.DestroyShader(pixel);
    return ok;
}

[[nodiscard]] int Channel(const u8 (&pixels)[kPixelBytes], u32 x, u32 y, u32 channel) noexcept
{
    return pixels[(y * kTarget + x) * 4 + channel];
}

[[nodiscard]] bool Near(int a, int b, int tolerance) noexcept
{
    return a - b <= tolerance && b - a <= tolerance;
}

}  // namespace

CG_TEST(gpu_parity_compute_with_bind_groups_and_push_constants)
{
    static u32 results[2][kElements];
    bool ran[2] = {false, false};
    const Backend* backends[2] = {D3D12Backend(), VulkanBackend()};
    for (u32 index = 0; index < 2; ++index)
    {
        Gpu gpu;
        if (!Open(backends[index], gpu))
        {
            continue;
        }
        std::printf("  %s: %s\n", gpu.device.BackendName(), gpu.device.Capabilities().adapter.name);
        CG_REQUIRE(RunCompute(gpu, results[index]));
        ran[index] = true;
        u32 wrong = 0;
        for (u32 element = 0; element < kElements; ++element)
        {
            wrong += results[index][element] != (element * 3U + 1U) * 3U + 1U ? 1U : 0U;
        }
        CG_EXPECT_EQ(wrong, u32{0});
        ReleaseFence(gpu);
        std::printf("  leaks after run: %u\n", gpu.device.ReportLeaks());
        CG_EXPECT_EQ(gpu.device.ReportLeaks(), u32{0});
    }
    if (ran[0] && ran[1])
    {
        // The two backends computed the same buffer, byte for byte.
        CG_EXPECT_TRUE(std::memcmp(results[0], results[1], sizeof(results[0])) == 0);
    }
}

CG_TEST(gpu_parity_graphics_render_pass_and_draw)
{
    static u8 images[2][kPixelBytes];
    bool ran[2] = {false, false};
    const Backend* backends[2] = {D3D12Backend(), VulkanBackend()};
    for (u32 index = 0; index < 2; ++index)
    {
        Gpu gpu;
        if (!Open(backends[index], gpu))
        {
            continue;
        }
        std::printf("  %s: %s\n", gpu.device.BackendName(), gpu.device.Capabilities().adapter.name);
        CG_REQUIRE(RunGraphics(gpu, images[index]));
        ran[index] = true;
        const auto& pixels = images[index];
        // Clip-space y points up on both backends: the red corner of the triangle
        // is the bottom-left of the target, the blue one the top-left.
        CG_EXPECT_TRUE(Channel(pixels, 0, kTarget - 1, 0) > 240);
        CG_EXPECT_TRUE(Channel(pixels, 0, kTarget - 1, 1) < 12);
        CG_EXPECT_TRUE(Channel(pixels, 0, kTarget - 1, 2) < 12);
        // Barycentric weights at the centre are about 1/2, 1/4 and 1/4.
        CG_EXPECT_TRUE(Near(Channel(pixels, 32, 32, 0), 127, 8));
        CG_EXPECT_TRUE(Near(Channel(pixels, 32, 32, 1), 65, 8));
        CG_EXPECT_TRUE(Near(Channel(pixels, 32, 32, 2), 63, 8));
        CG_EXPECT_EQ(Channel(pixels, 32, 32, 3), 255);
        ReleaseFence(gpu);
        std::printf("  leaks after run: %u\n", gpu.device.ReportLeaks());
        CG_EXPECT_EQ(gpu.device.ReportLeaks(), u32{0});
    }
    if (ran[0] && ran[1])
    {
        // Same renderer code, same picture: no channel of any pixel differs by more
        // than interpolation rounding.
        u32 outliers = 0;
        for (u32 index = 0; index < kPixelBytes; ++index)
        {
            outliers += Near(images[0][index], images[1][index], 3) ? 0U : 1U;
        }
        CG_EXPECT_EQ(outliers, u32{0});
    }
}

CG_TEST(gpu_parity_swapchain_presents_frames)
{
    if (!cyber::platform::IsWindowSystemSupported())
    {
        std::printf("  skipped: no window system\n");
        return;
    }
    const Backend* backends[2] = {D3D12Backend(), VulkanBackend()};
    for (u32 index = 0; index < 2; ++index)
    {
        Gpu gpu;
        if (!Open(backends[index], gpu))
        {
            continue;
        }
        cyber::platform::WindowDesc windowDesc{};
        windowDesc.title = "swapchain parity";
        windowDesc.width = 320;
        windowDesc.height = 240;
        windowDesc.visible = false;
        auto window = cyber::platform::Window::Create(windowDesc, gpu.heap);
        if (!window)
        {
            std::printf("  skipped: window creation failed\n");
            continue;
        }
        SwapChainDesc desc{};
        desc.surface.nativeWindow = window.Value().NativeHandle();
        desc.surface.width = 320;
        desc.surface.height = 240;
        desc.format = Format::BGRA8UnormSrgb;
        auto swapchain = gpu.device.CreateSwapChain(desc);
        if (!swapchain)
        {
            std::printf("  %s swapchain skipped: %s\n", gpu.device.BackendName(),
                        swapchain.Error().message);
            continue;
        }
        std::printf("  %s: swapchain %ux%u\n", gpu.device.BackendName(),
                    swapchain.Value().Info().width, swapchain.Value().Info().height);
        CG_EXPECT_TRUE(swapchain.Value().Info().imageCount >= 2);

        CommandListDesc listDesc{};
        auto created = gpu.device.CreateCommandList(listDesc);
        CG_REQUIRE(created);
        CommandList list = std::move(created.Value());
        for (u32 frame = 0; frame < 5; ++frame)
        {
            auto acquired = swapchain.Value().AcquireNextImage();
            CG_REQUIRE(acquired);
            const SwapChainFrame image = acquired.Value();
            CG_REQUIRE(image.valid);
            list.Reset();
            CG_REQUIRE(list.Begin().IsOk());
            SubresourceRange whole{};
            whole.mipLevelCount = 1;
            whole.layerCount = 1;
            // A back buffer starts in `Present` on both backends.
            const TextureBarrier toTarget[1] = {
                {image.texture, whole, ResourceState::Present, ResourceState::RenderTarget}};
            CG_REQUIRE(list.Barrier({}, Span<const TextureBarrier>(toTarget, 1)).IsOk());
            ColorAttachment color{};
            color.view = image.view;
            color.loadOp = LoadOp::Clear;
            color.storeOp = StoreOp::Store;
            color.clearValue = ClearColorValue{0.2F, 0.4F, 0.8F, 1.0F};
            RenderPassDesc pass{};
            pass.colorAttachments = Span<const ColorAttachment>(&color, 1);
            CG_REQUIRE(list.BeginRenderPass(pass).IsOk());
            CG_REQUIRE(list.EndRenderPass().IsOk());
            const TextureBarrier toPresent[1] = {
                {image.texture, whole, ResourceState::RenderTarget, ResourceState::Present}};
            CG_REQUIRE(list.Barrier({}, Span<const TextureBarrier>(toPresent, 1)).IsOk());
            CG_REQUIRE(list.End().IsOk());
            CG_REQUIRE(SubmitAndWait(gpu, list));
            CG_REQUIRE(swapchain.Value().Present().IsOk());
        }
        CG_EXPECT_TRUE(gpu.device.WaitIdle().IsOk());
    }
}
