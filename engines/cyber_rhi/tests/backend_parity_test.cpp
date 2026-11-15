// The same RHI calls against every compiled-in hardware backend. Phase 34's done
// line is "the same renderer code on D3D12 and Vulkan"; this is the first place
// that is checked: one body, run once per backend, with the behaviour the
// contract fixes asserted identically and the differences (what Vulkan does not
// implement yet) asserted separately and by name.

#include "../src/vulkan/spirv_builder.h"

#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/rhi/backend.h>
#include <cyber/rhi/d3d12.h>
#include <cyber/rhi/device.h>
#include <cyber/rhi/vulkan.h>
#include <cyber/test/test.h>

#include <cstdio>
#include <cstring>
#include <utility>

using namespace cyber::rhi;
using cyber::core::ErrorCode;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;
using cyber::memory::Byte;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;

namespace
{

// Opens adapter 0 of `backend` with validation, or reports why the machine cannot
// run the test. A backend with no adapter is skipped: the machine is the variable.
[[nodiscard]] bool OpenDevice(const Backend* backend, HeapAllocator& heap, Device& out) noexcept
{
    if (backend == nullptr || backend->enumerateAdapters(nullptr, 0) == 0)
    {
        std::printf("  skipped: no adapter\n");
        return false;
    }
    DeviceDesc desc{};
    desc.backend = backend;
    desc.adapterIndex = 0;
    desc.validation = ValidationLevel::Standard;
    auto created = Device::Create(desc, heap);
    if (!created)
    {
        std::printf("  skipped: %s\n", created.Error().message);
        return false;
    }
    out = std::move(created.Value());
    return true;
}

template <typename Handle>
[[nodiscard]] ObjectRef RefOf(ObjectKind kind, Handle handle) noexcept
{
    ObjectRef ref{};
    ref.kind = kind;
    ref.packed = PackHandle(handle);
    return ref;
}

void CheckBuffers(Device& device)
{
    BufferDesc upload{};
    upload.usage = BufferUsage::HostVisible | BufferUsage::CopySource;
    upload.sizeBytes = 4096;
    upload.label = "parity upload";
    auto uploadBuffer = device.CreateBuffer(upload);
    CG_REQUIRE(uploadBuffer);

    BufferDesc readback{};
    readback.usage = BufferUsage::Readback | BufferUsage::CopyDestination;
    readback.sizeBytes = 4096;
    auto readbackBuffer = device.CreateBuffer(readback);
    CG_REQUIRE(readbackBuffer);

    BufferDesc local{};
    local.usage = BufferUsage::Vertex | BufferUsage::Storage | BufferUsage::CopyDestination;
    local.sizeBytes = 65536;
    local.elementStride = 16;
    auto localBuffer = device.CreateBuffer(local);
    CG_REQUIRE(localBuffer);

    const BufferInfo info = device.BufferInfoOf(localBuffer.Value());
    CG_EXPECT_TRUE(info.valid);
    CG_EXPECT_EQ(info.sizeBytes, u64{65536});
    CG_EXPECT_EQ(info.elementStride, u32{16});

    // Host access follows the usage: HostVisible is writable, Readback readable,
    // and nothing else is either.
    u8 data[256];
    for (u32 index = 0; index < 256; ++index)
    {
        data[index] = static_cast<u8>(index);
    }
    const Span<const Byte> bytes(reinterpret_cast<const Byte*>(data), sizeof(data));
    CG_EXPECT_TRUE(device.WriteBuffer(uploadBuffer.Value(), 128, bytes).IsOk());
    CG_EXPECT_FALSE(device.WriteBuffer(localBuffer.Value(), 0, bytes).IsOk());
    CG_EXPECT_FALSE(device.WriteBuffer(uploadBuffer.Value(), 4000, bytes).IsOk());  // past the end
    u8 sink[16];
    CG_EXPECT_FALSE(
        device.ReadBuffer(uploadBuffer.Value(), 0, Span<Byte>(reinterpret_cast<Byte*>(sink), 16))
            .IsOk());
    CG_EXPECT_TRUE(
        device.ReadBuffer(readbackBuffer.Value(), 0, Span<Byte>(reinterpret_cast<Byte*>(sink), 16))
            .IsOk());

    // The label a caller gave is what the object reports back.
    const ObjectNameInfo name =
        device.ObjectName(RefOf(ObjectKind::Buffer, uploadBuffer.Value()));
    CG_EXPECT_TRUE(name.named);
    CG_EXPECT_STREQ(name.name, "parity upload");
    CG_EXPECT_FALSE(device.ObjectName(RefOf(ObjectKind::Buffer, localBuffer.Value())).named);

    CG_EXPECT_TRUE(device.DestroyBuffer(uploadBuffer.Value()).IsOk());
    CG_EXPECT_TRUE(device.DestroyBuffer(readbackBuffer.Value()).IsOk());
    CG_EXPECT_TRUE(device.DestroyBuffer(localBuffer.Value()).IsOk());
    // A destroyed handle is stale, not a crash and not a second destroy.
    CG_EXPECT_FALSE(device.DestroyBuffer(localBuffer.Value()).IsOk());
    CG_EXPECT_FALSE(device.BufferInfoOf(localBuffer.Value()).valid);
}

void CheckTextures(Device& device)
{
    TextureDesc color{};
    color.format = Format::RGBA8UnormSrgb;
    color.usage = TextureUsage::Sampled | TextureUsage::CopyDestination;
    color.width = 256;
    color.height = 256;
    color.mipLevels = 9;
    color.arrayLayers = 2;
    auto texture = device.CreateTexture(color);
    CG_REQUIRE(texture);

    const TextureInfo info = device.TextureInfoOf(texture.Value());
    CG_EXPECT_TRUE(info.valid);
    CG_EXPECT_EQ(info.width, u32{256});
    CG_EXPECT_EQ(info.mipLevels, u32{9});
    CG_EXPECT_EQ(info.arrayLayers, u32{2});
    CG_EXPECT_TRUE(info.sizeBytes >= 256ULL * 256ULL * 4ULL);
    CG_EXPECT_TRUE(device.TextureStateOf(texture.Value(), 0, 0) == ResourceState::Undefined);

    // Views: the whole array, one mip of one layer, and a range outside the texture.
    TextureViewDesc whole{};
    whole.texture = texture.Value();
    whole.format = Format::RGBA8UnormSrgb;
    whole.range.mipLevelCount = 9;
    whole.range.layerCount = 2;
    whole.label = "parity view";
    auto wholeView = device.CreateTextureView(whole);
    CG_REQUIRE(wholeView);
    TextureViewDesc slice = whole;
    slice.range.baseMipLevel = 3;
    slice.range.mipLevelCount = 2;
    slice.range.baseArrayLayer = 1;
    slice.range.layerCount = 1;
    slice.label = nullptr;
    auto sliceView = device.CreateTextureView(slice);
    CG_REQUIRE(sliceView);
    TextureViewDesc outside = whole;
    outside.range.baseMipLevel = 8;
    outside.range.mipLevelCount = 4;
    CG_EXPECT_FALSE(device.CreateTextureView(outside).IsOk());

    const TextureViewInfo viewInfo = device.TextureViewInfoOf(sliceView.Value());
    CG_EXPECT_TRUE(viewInfo.valid);
    CG_EXPECT_EQ(viewInfo.range.baseMipLevel, u32{3});
    CG_EXPECT_EQ(viewInfo.range.mipLevelCount, u32{2});

    // Depth, 3D and cube textures.
    TextureDesc depth{};
    depth.format = Format::D32Float;
    depth.usage = TextureUsage::DepthStencil;
    depth.width = 128;
    depth.height = 128;
    auto depthTexture = device.CreateTexture(depth);
    CG_EXPECT_TRUE(depthTexture.IsOk());

    TextureDesc volume{};
    volume.dimension = TextureDimension::Dim3D;
    volume.format = Format::RGBA16Float;
    volume.usage = TextureUsage::Sampled | TextureUsage::Storage;
    volume.width = 32;
    volume.height = 32;
    volume.depth = 32;
    auto volumeTexture = device.CreateTexture(volume);
    CG_EXPECT_TRUE(volumeTexture.IsOk());

    TextureDesc cube{};
    cube.dimension = TextureDimension::Cube;
    cube.format = Format::RGBA8Unorm;
    cube.usage = TextureUsage::Sampled | TextureUsage::RenderTarget;
    cube.width = 64;
    cube.height = 64;
    cube.arrayLayers = 6;
    auto cubeTexture = device.CreateTexture(cube);
    CG_EXPECT_TRUE(cubeTexture.IsOk());

    // A texture with no extent or no usage is refused.
    TextureDesc bad{};
    bad.usage = TextureUsage::Sampled;
    CG_EXPECT_FALSE(device.CreateTexture(bad).IsOk());

    CG_EXPECT_TRUE(device.DestroyTextureView(sliceView.Value()).IsOk());
    CG_EXPECT_TRUE(device.DestroyTextureView(wholeView.Value()).IsOk());
    CG_EXPECT_TRUE(device.DestroyTexture(texture.Value()).IsOk());
    if (depthTexture)
    {
        CG_EXPECT_TRUE(device.DestroyTexture(depthTexture.Value()).IsOk());
    }
    if (volumeTexture)
    {
        CG_EXPECT_TRUE(device.DestroyTexture(volumeTexture.Value()).IsOk());
    }
    if (cubeTexture)
    {
        CG_EXPECT_TRUE(device.DestroyTexture(cubeTexture.Value()).IsOk());
    }
    CG_EXPECT_FALSE(device.TextureInfoOf(texture.Value()).valid);
}

void CheckSamplers(Device& device)
{
    SamplerDesc linear{};
    linear.label = "parity sampler";
    auto sampler = device.CreateSampler(linear);
    CG_REQUIRE(sampler);

    SamplerDesc shadow{};
    shadow.compareEnabled = true;
    shadow.compareFunction = CompareFunction::GreaterEqual;
    shadow.addressU = AddressMode::ClampToBorder;
    shadow.addressV = AddressMode::ClampToBorder;
    shadow.borderColor = BorderColor::OpaqueWhite;
    auto shadowSampler = device.CreateSampler(shadow);
    CG_EXPECT_TRUE(shadowSampler.IsOk());

    SamplerDesc anisotropic{};
    anisotropic.maxAnisotropy = 8;
    auto anisotropicSampler = device.CreateSampler(anisotropic);
    CG_EXPECT_TRUE(anisotropicSampler.IsOk());

    CG_EXPECT_TRUE(device.DestroySampler(sampler.Value()).IsOk());
    if (shadowSampler)
    {
        CG_EXPECT_TRUE(device.DestroySampler(shadowSampler.Value()).IsOk());
    }
    if (anisotropicSampler)
    {
        CG_EXPECT_TRUE(device.DestroySampler(anisotropicSampler.Value()).IsOk());
    }
    CG_EXPECT_FALSE(device.DestroySampler(sampler.Value()).IsOk());
}

void CheckFences(Device& device)
{
    FenceDesc desc{};
    desc.initialValue = 3;
    desc.label = "parity fence";
    auto fence = device.CreateFence(desc);
    CG_REQUIRE(fence);
    CG_EXPECT_EQ(device.FenceValue(fence.Value()), u64{3});

    CG_EXPECT_TRUE(device.SignalFence(fence.Value(), 5).IsOk());
    CG_EXPECT_EQ(device.FenceValue(fence.Value()), u64{5});
    // A lower signal is already satisfied and never moves the value back.
    CG_EXPECT_TRUE(device.SignalFence(fence.Value(), 4).IsOk());
    CG_EXPECT_EQ(device.FenceValue(fence.Value()), u64{5});

    // Reached values return at once; an unreached one times out with the code the
    // contract fixes, and a zero timeout is a poll.
    CG_EXPECT_TRUE(device.WaitFence(fence.Value(), 5, 0).IsOk());
    const auto timedOut = device.WaitFence(fence.Value(), 9, 1000000ULL);
    CG_REQUIRE(!timedOut);
    CG_EXPECT_TRUE(timedOut.Error().code == ErrorCode::NotFound);
    CG_EXPECT_FALSE(device.WaitFence(fence.Value(), 9, 0).IsOk());

    CG_EXPECT_TRUE(device.ObjectName(RefOf(ObjectKind::Fence, fence.Value())).named);
    CG_EXPECT_TRUE(device.DestroyFence(fence.Value()).IsOk());
    CG_EXPECT_FALSE(device.DestroyFence(fence.Value()).IsOk());
}

void CheckDeviceReport(Device& device, BackendKind kind)
{
    const Capabilities capabilities = device.Capabilities();
    CG_EXPECT_TRUE(capabilities.adapter.backend == kind);
    CG_EXPECT_TRUE(capabilities.adapter.name[0] != '\0');
    CG_EXPECT_TRUE(capabilities.features.Has(Feature::TimelineFences));
    CG_EXPECT_TRUE(capabilities.limits.maxTexture2D >= 4096);
    CG_EXPECT_TRUE(capabilities.limits.maxPushConstantBytes >= 128);
    CG_EXPECT_TRUE(capabilities.limits.maxColorAttachments >= 4);
    CG_EXPECT_TRUE(device.Queue(QueueKind::Graphics).IsValid());
    CG_EXPECT_TRUE(device.Queue(QueueKind::Compute).IsValid());
    CG_EXPECT_TRUE(device.Queue(QueueKind::Copy).IsValid());
    CG_EXPECT_TRUE(device.WaitIdle().IsOk());
}

void RunParity(const Backend* backend, BackendKind kind)
{
    HeapAllocator heap;
    Device device;
    if (!OpenDevice(backend, heap, device))
    {
        return;
    }
    std::printf("  backend %s on %s\n", device.BackendName(), device.Capabilities().adapter.name);
    CheckDeviceReport(device, kind);
    CheckBuffers(device);
    CheckTextures(device);
    CheckSamplers(device);
    CheckFences(device);
    // Everything created above was destroyed above.
    CG_EXPECT_EQ(device.ReportLeaks(), u32{0});

    // Memory is reported for the buffers and textures created, and returns to
    // zero used bytes.
    const MemoryStats stats = device.MemoryStats();
    CG_EXPECT_EQ(stats.UsedBytes(), u64{0});
}

}  // namespace

CG_TEST(backend_parity_d3d12)
{
    RunParity(D3D12Backend(), BackendKind::D3D12);
}

CG_TEST(backend_parity_vulkan)
{
    RunParity(VulkanBackend(), BackendKind::Vulkan);
}

CG_TEST(backend_parity_leaks_are_reported_by_both)
{
    for (const Backend* backend : {D3D12Backend(), VulkanBackend()})
    {
        HeapAllocator heap;
        Device device;
        if (!OpenDevice(backend, heap, device))
        {
            continue;
        }
        BufferDesc desc{};
        desc.usage = BufferUsage::CopySource | BufferUsage::CopyDestination;
        desc.sizeBytes = 1024;
        auto buffer = device.CreateBuffer(desc);
        CG_REQUIRE(buffer);
        CG_EXPECT_EQ(device.ReportLeaks(), u32{1});
        // The device destroys what is left at teardown without crashing.
    }
}

CG_TEST(vulkan_device_accepts_spirv_shaders)
{
    HeapAllocator heap;
    Device device;
    if (!OpenDevice(VulkanBackend(), heap, device))
    {
        return;
    }
    // The smallest valid compute shader: an empty `main`.
    using namespace cyber::rhi::detail;
    cyber::memory::DynamicArray<u32> words(heap);
    {
        SpirvBuilder b(heap);
        const u32 voidT = b.NewId();
        const u32 fnT = b.NewId();
        const u32 mainFn = b.NewId();
        const u32 entry = b.NewId();
        b.Capability(spv::CapabilityShader);
        b.MemoryModelGlsl450();
        b.EntryPoint(spv::ExecutionGLCompute, mainFn, "main", nullptr, 0);
        b.LocalSize(mainFn, 1, 1, 1);
        b.Declare(spv::OpTypeVoid, {voidT});
        b.Declare(spv::OpTypeFunction, {fnT, voidT});
        b.Code(spv::OpFunction, {voidT, mainFn, 0, fnT});
        b.Code(spv::OpLabel, {entry});
        b.Code(spv::OpReturn, {});
        b.Code(spv::OpFunctionEnd, {});
        CG_REQUIRE(b.Finish(words).IsOk());
    }
    ShaderDesc desc{};
    desc.stage = ShaderStage::Compute;
    desc.format = ShaderBytecodeFormat::SpirV;
    desc.code = Span<const Byte>(reinterpret_cast<const Byte*>(words.Data()), words.Size() * 4);
    desc.entryPoint = "main";
    desc.label = "empty compute";
    auto shader = device.CreateShader(desc);
    CG_REQUIRE(shader);
    const ShaderInfo info = device.ShaderInfoOf(shader.Value());
    CG_EXPECT_TRUE(info.valid);
    CG_EXPECT_TRUE(info.stage == ShaderStage::Compute);
    CG_EXPECT_TRUE(info.format == ShaderBytecodeFormat::SpirV);
    CG_EXPECT_EQ(info.codeSize, static_cast<u64>(words.Size() * 4));
    CG_EXPECT_TRUE(device.ObjectName(RefOf(ObjectKind::Shader, shader.Value())).named);
    CG_EXPECT_TRUE(device.DestroyShader(shader.Value()).IsOk());
    CG_EXPECT_EQ(device.ReportLeaks(), u32{0});
}

CG_TEST(vulkan_device_names_views_and_samplers)
{
    // Unlike D3D12, whose views and samplers are descriptors with no driver object
    // to name, Vulkan names both, so the label a caller gave reads back.
    HeapAllocator heap;
    Device device;
    if (!OpenDevice(VulkanBackend(), heap, device))
    {
        return;
    }
    TextureDesc desc{};
    desc.usage = TextureUsage::Sampled;
    desc.width = 16;
    desc.height = 16;
    auto texture = device.CreateTexture(desc);
    CG_REQUIRE(texture);
    TextureViewDesc viewDesc{};
    viewDesc.texture = texture.Value();
    viewDesc.range.mipLevelCount = 1;
    viewDesc.range.layerCount = 1;
    viewDesc.label = "named view";
    auto view = device.CreateTextureView(viewDesc);
    CG_REQUIRE(view);
    const ObjectNameInfo viewName = device.ObjectName(RefOf(ObjectKind::TextureView, view.Value()));
    CG_EXPECT_TRUE(viewName.named);
    CG_EXPECT_STREQ(viewName.name, "named view");
    SamplerDesc samplerDesc{};
    samplerDesc.label = "named sampler";
    auto sampler = device.CreateSampler(samplerDesc);
    CG_REQUIRE(sampler);
    const ObjectNameInfo samplerName =
        device.ObjectName(RefOf(ObjectKind::Sampler, sampler.Value()));
    CG_EXPECT_TRUE(samplerName.named);
    CG_EXPECT_STREQ(samplerName.name, "named sampler");
    CG_EXPECT_TRUE(device.DestroySampler(sampler.Value()).IsOk());
    CG_EXPECT_TRUE(device.DestroyTextureView(view.Value()).IsOk());
    CG_EXPECT_TRUE(device.DestroyTexture(texture.Value()).IsOk());
}
