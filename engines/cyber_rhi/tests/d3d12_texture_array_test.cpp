// Arrays of sampled texture views (`Texture2D Tex[N]`) through the Direct3D 12
// backend. Each variant binds six distinct textures as one array and has a
// compute shader read one texel of each; the answer is known exactly, so a
// wrong element, a stale descriptor or a misplaced range shows up as a
// mismatch. Run on every adapter, the hardware one and WARP.

#include <cyber/memory/heap_allocator.h>
#include <cyber/rhi/backend.h>
#include <cyber/rhi/binding.h>
#include <cyber/rhi/buffer.h>
#include <cyber/rhi/command_list.h>
#include <cyber/rhi/d3d12.h>
#include <cyber/rhi/device.h>
#include <cyber/rhi/pipeline.h>
#include <cyber/rhi/resource_state.h>
#include <cyber/rhi/shader.h>
#include <cyber/rhi/submission.h>
#include <cyber/test/test.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <dxcapi.h>
#include <functional>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace
{

using cyber::core::Byte;
using cyber::core::u32;
using cyber::core::u64;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
namespace rhi = cyber::rhi;

constexpr u32 kTextures = 6;
constexpr u32 kThreads = 64;

// Texture i holds the value 10 + 20 * i in every texel's red channel.
u32 ExpectedRed(u32 texture) noexcept
{
    return 10u + 20u * texture;
}

struct Variant
{
    const char* name;
    const char* body;
};

const char* const kHeader = R"hlsl(
Texture2D<float4> Tex[6] : register(t0, space0);
RWStructuredBuffer<uint> Output : register(u0, space0);
uint Red(float4 c) { return (uint)round(saturate(c.x) * 255.0f); }
)hlsl";

const Variant kVariants[] = {
    // A literal index per access.
    {"literal",
     R"hlsl(
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    Output[0u * 64u + id.x] = Red(Tex[0].Load(int3(0, 0, 0)));
    Output[1u * 64u + id.x] = Red(Tex[1].Load(int3(0, 0, 0)));
    Output[2u * 64u + id.x] = Red(Tex[2].Load(int3(0, 0, 0)));
    Output[3u * 64u + id.x] = Red(Tex[3].Load(int3(0, 0, 0)));
    Output[4u * 64u + id.x] = Red(Tex[4].Load(int3(0, 0, 0)));
    Output[5u * 64u + id.x] = Red(Tex[5].Load(int3(0, 0, 0)));
}
)hlsl"},
    // A loop the compiler can unroll.
    {"unrolled",
     R"hlsl(
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    [unroll]
    for (uint t = 0; t < 6; ++t)
    {
        Output[t * 64u + id.x] = Red(Tex[t].Load(int3(0, 0, 0)));
    }
}
)hlsl"},
    // A loop kept as a loop, so the index is a run-time value.
    {"dynamic_uniform",
     R"hlsl(
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    [loop]
    for (uint t = 0; t < 6; ++t)
    {
        Output[t * 64u + id.x] = Red(Tex[t].Load(int3(0, 0, 0)));
    }
}
)hlsl"},
    // An index that differs per thread.
    {"non_uniform",
     R"hlsl(
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint t = id.x % 6u;
    Output[id.x] = Red(Tex[NonUniformResourceIndex(t)].Load(int3(0, 0, 0)));
}
)hlsl"},
};

// Compiles HLSL to DXIL through the DirectX Shader Compiler when its DLL can be
// found (next to the executable, on PATH, or in a Windows SDK). An empty result
// means the compiler is not installed and the DXIL variants are skipped.
std::vector<Byte> CompileDxil(const std::string& hlsl)
{
    HMODULE module = LoadLibraryW(L"dxcompiler.dll");
    if (module == nullptr)
    {
        // The newest Windows SDK's `bin\<version>\x64` holds a current compiler
        // with the DLL that signs the container beside it.
        wchar_t base[MAX_PATH]{};
        const DWORD length = GetEnvironmentVariableW(L"ProgramFiles(x86)", base, MAX_PATH);
        if (length > 0 && length < MAX_PATH)
        {
            const std::wstring bin = std::wstring(base) + L"\\Windows Kits\\10\\bin\\";
            std::vector<std::wstring> versions;
            WIN32_FIND_DATAW found{};
            const HANDLE search = FindFirstFileW((bin + L"10.*").c_str(), &found);
            if (search != INVALID_HANDLE_VALUE)
            {
                do
                {
                    versions.push_back(found.cFileName);
                } while (FindNextFileW(search, &found) != 0);
                FindClose(search);
            }
            std::sort(versions.begin(), versions.end(), std::greater<std::wstring>());
            for (const std::wstring& version : versions)
            {
                const std::wstring root = bin + version + L"\\x64\\";
                (void)LoadLibraryW((root + L"dxil.dll").c_str());
                module = LoadLibraryW((root + L"dxcompiler.dll").c_str());
                if (module != nullptr)
                {
                    break;
                }
            }
        }
    }
    if (module == nullptr)
    {
        std::printf("dxc: dxcompiler.dll not found (error %lu)\n", GetLastError());
        return {};
    }
    const auto create =
        reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(module, "DxcCreateInstance"));
    if (create == nullptr)
    {
        return {};
    }
    IDxcUtils* utils = nullptr;
    IDxcCompiler3* compiler = nullptr;
    std::vector<Byte> result;
    const HRESULT utilsResult = create(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
    const HRESULT compilerResult = create(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));
    if (SUCCEEDED(utilsResult) && SUCCEEDED(compilerResult))
    {
        DxcBuffer buffer{hlsl.data(), hlsl.size(), DXC_CP_UTF8};
        LPCWSTR arguments[] = {L"-E", L"main", L"-T", L"cs_6_0"};
        IDxcResult* compiled = nullptr;
        if (SUCCEEDED(compiler->Compile(&buffer, arguments, 4, nullptr, IID_PPV_ARGS(&compiled))))
        {
            HRESULT status = E_FAIL;
            compiled->GetStatus(&status);
            if (FAILED(status))
            {
                IDxcBlobUtf8* messages = nullptr;
                if (SUCCEEDED(
                        compiled->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&messages), nullptr)) &&
                    messages != nullptr)
                {
                    std::printf("dxc: %s\n", messages->GetStringPointer());
                    messages->Release();
                }
            }
            IDxcBlob* object = nullptr;
            if (SUCCEEDED(status) &&
                SUCCEEDED(compiled->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object), nullptr)) &&
                object != nullptr)
            {
                const auto* bytes = static_cast<const Byte*>(object->GetBufferPointer());
                result.assign(bytes, bytes + object->GetBufferSize());
                object->Release();
            }
            compiled->Release();
        }
    }
    if (compiler != nullptr)
    {
        compiler->Release();
    }
    if (utils != nullptr)
    {
        utils->Release();
    }
    return result;
}

template <class S>
bool Step(int line, const S& status)
{
    if (!status)
    {
        std::printf("rhi step at line %d failed: %s\n", line, status.Error().message);
    }
    return status.IsOk();
}

}  // namespace

CG_TEST(d3d12_texture_array_binding_selects_the_right_element)
{
    const rhi::Backend* backend = rhi::D3D12Backend();
    if (backend == nullptr)
    {
        CG_SKIP("no Direct3D 12 runtime");
    }
    rhi::AdapterInfo adapters[8]{};
    const u32 adapterCount = backend->enumerateAdapters(adapters, 8);
    u32 tested = 0;
    for (u32 adapterIndex = 0; adapterIndex < adapterCount; ++adapterIndex)
    {
        HeapAllocator allocator;
        rhi::DeviceDesc deviceDesc{};
        deviceDesc.backend = backend;
        deviceDesc.adapterIndex = adapterIndex;
        deviceDesc.validation = rhi::ValidationLevel::Exhaustive;
        auto created = rhi::Device::Create(deviceDesc, allocator);
        if (!created)
        {
            continue;
        }
        rhi::Device device = std::move(created.Value());
        ++tested;

        // Six 4x4 textures, each uploaded with its own constant.
        rhi::TextureHandle textures[kTextures];
        rhi::TextureViewHandle views[kTextures];
        rhi::BufferDesc stageDesc{};
        stageDesc.usage = rhi::BufferUsage::HostVisible | rhi::BufferUsage::CopySource;
        stageDesc.sizeBytes = 256u * 4u;
        stageDesc.label = "array_upload";
        auto stage = device.CreateBuffer(stageDesc);
        CG_REQUIRE(stage.IsOk());
        std::vector<Byte> pixels(256u * 4u);
        for (u32 row = 0; row < 4; ++row)
        {
            for (u32 col = 0; col < 4; ++col)
            {
                pixels[row * 256u + col * 4u + 3u] = Byte{255};
            }
        }
        for (u32 i = 0; i < kTextures; ++i)
        {
            for (u32 row = 0; row < 4; ++row)
            {
                for (u32 col = 0; col < 4; ++col)
                {
                    pixels[row * 256u + col * 4u] = static_cast<Byte>(ExpectedRed(i));
                }
            }
            rhi::TextureDesc textureDesc{};
            textureDesc.format = rhi::Format::RGBA8Unorm;
            textureDesc.usage = rhi::TextureUsage::Sampled | rhi::TextureUsage::CopyDestination;
            textureDesc.width = 4;
            textureDesc.height = 4;
            auto texture = device.CreateTexture(textureDesc);
            CG_REQUIRE(texture.IsOk());
            textures[i] = texture.Value();
            rhi::TextureViewDesc viewDesc{};
            viewDesc.texture = textures[i];
            viewDesc.format = rhi::Format::RGBA8Unorm;
            viewDesc.range.mipLevelCount = 1;
            auto view = device.CreateTextureView(viewDesc);
            CG_REQUIRE(view.IsOk());
            views[i] = view.Value();

            CG_REQUIRE(
                Step(__LINE__, device.WriteBuffer(stage.Value(), 0,
                                                  Span<const Byte>(pixels.data(), pixels.size()))));
            rhi::CommandListDesc listDesc{};
            listDesc.queue = rhi::QueueKind::Graphics;
            auto createdList = device.CreateCommandList(listDesc);
            CG_REQUIRE(createdList.IsOk());
            rhi::CommandList list = std::move(createdList.Value());
            CG_REQUIRE(Step(__LINE__, list.Begin()));
            rhi::SubresourceRange range{};
            range.mipLevelCount = 1;
            range.layerCount = 1;
            const rhi::BufferBarrier bufferBarrier{
                stage.Value(), device.BufferStateOf(stage.Value()), rhi::ResourceState::CopySource};
            const rhi::TextureBarrier toCopy{textures[i], range,
                                             device.TextureStateOf(textures[i], 0, 0),
                                             rhi::ResourceState::CopyDestination};
            const bool bufferReady = bufferBarrier.from == rhi::ResourceState::CopySource;
            CG_REQUIRE(Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(
                                                       &bufferBarrier, bufferReady ? 0u : 1u),
                                                   Span<const rhi::TextureBarrier>(&toCopy, 1))));
            rhi::BufferTextureCopy copy{};
            copy.buffer = stage.Value();
            copy.bufferRowPitch = 256;
            copy.texture = textures[i];
            copy.width = 4;
            copy.height = 4;
            CG_REQUIRE(Step(__LINE__, list.CopyBufferToTexture(copy)));
            const rhi::TextureBarrier toSample{textures[i], range,
                                               rhi::ResourceState::CopyDestination,
                                               rhi::ResourceState::ShaderRead};
            CG_REQUIRE(Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>{},
                                                   Span<const rhi::TextureBarrier>(&toSample, 1))));
            CG_REQUIRE(Step(__LINE__, list.End()));
            const rhi::CommandListHandle handle = list.Handle();
            rhi::SubmitDesc submit{};
            submit.queue = rhi::QueueKind::Graphics;
            submit.commandLists = Span<const rhi::CommandListHandle>(&handle, 1);
            CG_REQUIRE(Step(__LINE__, device.Submit(submit)));
            CG_REQUIRE(Step(__LINE__, device.WaitIdle()));
        }

        rhi::BufferDesc outDesc{};
        outDesc.usage = rhi::BufferUsage::Storage | rhi::BufferUsage::CopySource;
        outDesc.sizeBytes = static_cast<u64>(kTextures) * kThreads * sizeof(u32);
        outDesc.elementStride = 4;
        outDesc.label = "array_out";
        auto out = device.CreateBuffer(outDesc);
        CG_REQUIRE(out.IsOk());
        rhi::BufferDesc readDesc{};
        readDesc.usage = rhi::BufferUsage::Readback | rhi::BufferUsage::CopyDestination;
        readDesc.sizeBytes = outDesc.sizeBytes;
        readDesc.label = "array_readback";
        auto readback = device.CreateBuffer(readDesc);
        CG_REQUIRE(readback.IsOk());

        const rhi::ShaderStageMask compute = rhi::ShaderStageBit(rhi::ShaderStage::Compute);
        rhi::BindGroupLayoutEntry entries[2]{};
        entries[0] = {0, rhi::BindingType::SampledTexture, compute, kTextures, 0, false};
        entries[1] = {1, rhi::BindingType::StorageBuffer, compute, 1, 0, false};
        rhi::BindGroupLayoutDesc layoutDesc{};
        layoutDesc.entries = Span<const rhi::BindGroupLayoutEntry>(entries, 2);
        auto groupLayout = device.CreateBindGroupLayout(layoutDesc);
        CG_REQUIRE(groupLayout.IsOk());
        rhi::PipelineLayoutDesc pipelineLayoutDesc{};
        pipelineLayoutDesc.groups[static_cast<u32>(rhi::BindGroupSlot::PerFrame)] =
            groupLayout.Value();
        auto pipelineLayout = device.CreatePipelineLayout(pipelineLayoutDesc);
        CG_REQUIRE(pipelineLayout.IsOk());

        rhi::BindingResource resources[kTextures + 1]{};
        for (u32 i = 0; i < kTextures; ++i)
        {
            resources[i].type = rhi::BindingType::SampledTexture;
            resources[i].textureView = views[i];
        }
        resources[kTextures].type = rhi::BindingType::StorageBuffer;
        resources[kTextures].buffer = out.Value();
        rhi::BindGroupDesc groupDesc{};
        groupDesc.layout = groupLayout.Value();
        groupDesc.resources = Span<const rhi::BindingResource>(resources, kTextures + 1);
        auto group = device.CreateBindGroup(groupDesc);
        CG_REQUIRE(group.IsOk());

        for (u32 job = 0; job < static_cast<u32>(std::size(kVariants)) * 2u; ++job)
        {
            const Variant& variant = kVariants[job / 2u];
            const bool dxil = (job & 1u) != 0;
            const std::string hlsl = std::string(kHeader) + variant.body;
            std::vector<Byte> dxilBytes;
            if (dxil)
            {
                dxilBytes = CompileDxil(hlsl);
                if (dxilBytes.empty())
                {
                    std::printf("adapter %u variant %-16s (dxil) skipped: no dxcompiler.dll\n",
                                adapterIndex, variant.name);
                    continue;
                }
            }
            rhi::ShaderDesc shaderDesc{};
            shaderDesc.stage = rhi::ShaderStage::Compute;
            shaderDesc.format =
                dxil ? rhi::ShaderBytecodeFormat::Dxil : rhi::ShaderBytecodeFormat::HlslSource;
            shaderDesc.code =
                dxil ? Span<const Byte>(dxilBytes.data(), dxilBytes.size())
                     : Span<const Byte>(reinterpret_cast<const Byte*>(hlsl.data()), hlsl.size());
            shaderDesc.entryPoint = "main";
            shaderDesc.label = variant.name;
            auto shader = device.CreateShader(shaderDesc);
            CG_REQUIRE(shader.IsOk());
            rhi::ComputePipelineDesc pipelineDesc{};
            pipelineDesc.layout = pipelineLayout.Value();
            pipelineDesc.computeShader = shader.Value();
            auto pipeline = device.CreateComputePipeline(pipelineDesc);
            CG_REQUIRE(pipeline.IsOk());

            rhi::CommandListDesc listDesc{};
            listDesc.queue = rhi::QueueKind::Graphics;
            auto createdList = device.CreateCommandList(listDesc);
            CG_REQUIRE(createdList.IsOk());
            rhi::CommandList list = std::move(createdList.Value());
            CG_REQUIRE(Step(__LINE__, list.Begin()));
            const rhi::BufferBarrier toShader[] = {
                {out.Value(), device.BufferStateOf(out.Value()), rhi::ResourceState::ShaderWrite}};
            CG_REQUIRE(Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(toShader, 1))));
            CG_REQUIRE(Step(__LINE__, list.SetPipeline(pipeline.Value())));
            CG_REQUIRE(
                Step(__LINE__, list.SetBindGroup(rhi::BindGroupSlot::PerFrame, group.Value())));
            rhi::DispatchArguments dispatch{};
            dispatch.groupCountX = 1;
            CG_REQUIRE(Step(__LINE__, list.Dispatch(dispatch)));
            const rhi::BufferBarrier toRead[] = {
                {out.Value(), rhi::ResourceState::ShaderWrite, rhi::ResourceState::CopySource},
                {readback.Value(), device.BufferStateOf(readback.Value()),
                 rhi::ResourceState::CopyDestination}};
            CG_REQUIRE(Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(toRead, 2))));
            rhi::BufferCopy back{};
            back.source = out.Value();
            back.destination = readback.Value();
            back.sizeBytes = outDesc.sizeBytes;
            CG_REQUIRE(Step(__LINE__, list.CopyBuffer(back)));
            CG_REQUIRE(Step(__LINE__, list.End()));
            const rhi::CommandListHandle handle = list.Handle();
            rhi::SubmitDesc submit{};
            submit.queue = rhi::QueueKind::Graphics;
            submit.commandLists = Span<const rhi::CommandListHandle>(&handle, 1);
            CG_REQUIRE(Step(__LINE__, device.Submit(submit)));
            CG_REQUIRE(Step(__LINE__, device.WaitIdle()));
            u32 results[kTextures * kThreads] = {};
            CG_REQUIRE(Step(__LINE__, device.ReadBuffer(readback.Value(), 0,
                                                        Span<Byte>(reinterpret_cast<Byte*>(results),
                                                                   sizeof(results)))));

            u32 wrong = 0;
            const bool perThread = std::strcmp(variant.name, "non_uniform") == 0;
            for (u32 t = 0; t < kTextures; ++t)
            {
                for (u32 s = 0; s < kThreads; ++s)
                {
                    const u32 want = perThread ? ExpectedRed(s % kTextures) : ExpectedRed(t);
                    const u32 got = perThread ? results[s] : results[t * kThreads + s];
                    if (perThread && t != 0)
                    {
                        continue;
                    }
                    wrong += got != want ? 1u : 0u;
                }
            }
            std::printf("adapter %u variant %-16s %s wrong=%u\n", adapterIndex, variant.name,
                        dxil ? "dxil" : "dxbc", wrong);
            // Microsoft's software rasteriser (vendor 0x1414) returns element 0
            // for every constant index into a texture descriptor array. It is
            // the same for FXC bytecode (which is correct: `ld ... T0[3]`) and
            // for DXC's DXIL, so it is the adapter and not a compiler, and the
            // hardware adapter is right for all variants. Run-time and
            // non-uniform indices, which is how bindless access is written, are
            // right on both. A recorded driver defect, not an RHI one.
            const bool knownWarpDefect = adapters[adapterIndex].vendorId == 0x1414 &&
                                         (std::strcmp(variant.name, "literal") == 0 ||
                                          std::strcmp(variant.name, "unrolled") == 0);
            if (!knownWarpDefect)
            {
                CG_EXPECT_EQ(wrong, 0u);
            }
        }
    }
    if (tested == 0)
    {
        CG_SKIP("no Direct3D 12 adapter could be created");
    }
}
