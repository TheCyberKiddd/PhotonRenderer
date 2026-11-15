// GPU virtual texturing (Phase 33): the page table and cache atlas are mirrored
// into RHI textures and a compute shader samples them with the HLSL
// `GpuVirtualTexture::ShaderSource()` ships. On every Direct3D 12 adapter the
// shader's result for thousands of random (u, v, lod) samples is compared with
// `VirtualTexture::Sample`, both while pages are still loading (so lookups fall
// back to coarser ancestors) and once they are resident. The Null backend
// checks what `Sync` uploads and when.

#include <cyber/core/log.h>
#include <cyber/core/log_sinks.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/renderer/gpu_virtual_texture.h>
#include <cyber/renderer/virtual_texture.h>
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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using cyber::core::Byte;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;
using cyber::core::usize;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using namespace cyber::renderer;
namespace rhi = cyber::rhi;

constexpr u32 kSize = 4096;
constexpr u32 kPage = 128;
constexpr u32 kBorder = 4;

void TexelColour(u32 mip, u32 x, u32 y, u8 out[4])
{
    out[0] = static_cast<u8>((x * 2u) & 0xFFu);
    out[1] = static_cast<u8>((y * 2u + mip * 17u) & 0xFFu);
    out[2] = static_cast<u8>(mip * 40u);
    out[3] = 255;
}

class TestProducer final : public VirtualPageProducer
{
public:
    bool ProducePage(u32 mip, u32 pageX, u32 pageY, Span<Byte> rgba) noexcept override
    {
        const u32 stride = kPage + 2u * kBorder;
        const i32 extent = static_cast<i32>(kSize >> mip);
        for (u32 j = 0; j < stride; ++j)
        {
            for (u32 i = 0; i < stride; ++i)
            {
                const i32 gx = std::clamp(
                    static_cast<i32>(pageX * kPage + i) - static_cast<i32>(kBorder), 0, extent - 1);
                const i32 gy = std::clamp(
                    static_cast<i32>(pageY * kPage + j) - static_cast<i32>(kBorder), 0, extent - 1);
                u8 colour[4];
                TexelColour(mip, static_cast<u32>(gx), static_cast<u32>(gy), colour);
                for (u32 c = 0; c < 4; ++c)
                {
                    rgba[(static_cast<usize>(j) * stride + i) * 4u + c] =
                        static_cast<Byte>(colour[c]);
                }
            }
        }
        return true;
    }
};

VirtualTextureDesc Desc()
{
    VirtualTextureDesc desc;
    desc.width = kSize;
    desc.height = kSize;
    desc.pageSize = kPage;
    desc.border = kBorder;
    desc.cachePagesX = 8;
    desc.cachePagesY = 8;
    desc.ioThreads = 2;
    desc.maxInFlight = 16;
    desc.maxUploadsPerFrame = 8;
    return desc;
}

template <class S>
bool Step(int line, const S& status)
{
    if (!status)
    {
        std::printf("gpu step at line %d failed: %s\n", line, status.Error().message);
    }
    return status.IsOk();
}

template <class R>
bool StepR(int line, const R& result)
{
    if (!result)
    {
        std::printf("gpu step at line %d failed: %s\n", line, result.Error().message);
    }
    return result.IsOk();
}

struct Query
{
    f32 u, v, lod, unused;
};

std::vector<Query> MakeQueries(u32 count, u32 seed)
{
    std::vector<Query> queries(count);
    u32 state = seed;
    const auto next = [&state]() noexcept
    {
        state = state * 1664525u + 1013904223u;
        return static_cast<f32>(state >> 8u) * (1.0f / 16777216.0f);
    };
    for (Query& query : queries)
    {
        query.u = next();
        query.v = next();
        query.lod = next() * 5.5f;
        query.unused = 0.0f;
    }
    return queries;
}

// The resources the library's `VtSample` reads are declared before its source.
const char* const kTestGlobals = R"hlsl(
Texture2D<uint> g_vtPageTable : register(t0, space0);
Texture2D<float4> g_vtAtlas : register(t1, space0);
)hlsl";

const char* const kTestShader = R"hlsl(
StructuredBuffer<float4> Input : register(t2, space0);
StructuredBuffer<uint> Params : register(t3, space0);
RWStructuredBuffer<float4> Output : register(u0, space0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    VtParams p;
    p.width = Params[0];
    p.height = Params[1];
    p.pageSize = Params[2];
    p.border = Params[3];
    p.stride = Params[4];
    p.mipCount = Params[5];
    p.reserved0 = 0;
    p.reserved1 = 0;
    float4 q = Input[id.x];
    Output[id.x] = VtSample(p, q.xy, q.z);
}
)hlsl";

// Runs the test shader over `queries` on `device` and returns RGBA results.
bool SampleOnGpu(rhi::Device& device, const GpuVirtualTexture& gpu,
                 const std::vector<Query>& queries, std::vector<f32>& results)
{
    const u32 count = static_cast<u32>(queries.size());
    const GpuVirtualTextureParams params = gpu.Params();
    const u32 paramWords[8] = {params.width,
                               params.height,
                               params.pageSize,
                               params.border,
                               params.stride,
                               params.mipCount,
                               0u,
                               0u};

    const auto makeBuffer = [&device](rhi::BufferUsage usage, u64 bytes, u32 stride,
                                      const char* label) -> rhi::BufferHandle
    {
        rhi::BufferDesc desc{};
        desc.usage = usage;
        desc.sizeBytes = bytes;
        desc.elementStride = stride;
        desc.label = label;
        auto created = device.CreateBuffer(desc);
        return created ? created.Value() : rhi::BufferHandle{};
    };
    const u64 queryBytes = static_cast<u64>(count) * sizeof(Query);
    const u64 resultBytes = static_cast<u64>(count) * 16u;
    const rhi::BufferHandle queryStage =
        makeBuffer(rhi::BufferUsage::HostVisible | rhi::BufferUsage::CopySource, queryBytes, 0,
                   "vt_query_stage");
    const rhi::BufferHandle paramStage = makeBuffer(
        rhi::BufferUsage::HostVisible | rhi::BufferUsage::CopySource, 32, 0, "vt_param_stage");
    const rhi::BufferHandle queryBuffer =
        makeBuffer(rhi::BufferUsage::Storage | rhi::BufferUsage::CopyDestination, queryBytes, 16,
                   "vt_queries");
    const rhi::BufferHandle paramBuffer = makeBuffer(
        rhi::BufferUsage::Storage | rhi::BufferUsage::CopyDestination, 32, 4, "vt_params");
    const rhi::BufferHandle outBuffer = makeBuffer(
        rhi::BufferUsage::Storage | rhi::BufferUsage::CopySource, resultBytes, 16, "vt_results");
    const rhi::BufferHandle readback =
        makeBuffer(rhi::BufferUsage::Readback | rhi::BufferUsage::CopyDestination, resultBytes, 0,
                   "vt_readback");
    bool ok = queryStage.IsValid() && paramStage.IsValid() && queryBuffer.IsValid() &&
              paramBuffer.IsValid() && outBuffer.IsValid() && readback.IsValid();

    rhi::ShaderHandle shader{};
    rhi::BindGroupLayoutHandle groupLayout{};
    rhi::PipelineLayoutHandle pipelineLayout{};
    rhi::PipelineHandle pipeline{};
    rhi::BindGroupHandle group{};
    if (ok)
    {
        std::string source = kTestGlobals;
        source += GpuVirtualTexture::ShaderSource();
        source += kTestShader;
        rhi::ShaderDesc shaderDesc{};
        shaderDesc.stage = rhi::ShaderStage::Compute;
        shaderDesc.format = rhi::ShaderBytecodeFormat::HlslSource;
        shaderDesc.code =
            Span<const Byte>(reinterpret_cast<const Byte*>(source.data()), source.size());
        shaderDesc.entryPoint = "main";
        shaderDesc.label = "vt_test";
        auto createdShader = device.CreateShader(shaderDesc);
        if (!createdShader)
        {
            std::printf("shader: %s\n", createdShader.Error().message);
            ok = false;
        }
        else
        {
            shader = createdShader.Value();
        }
    }
    if (ok)
    {
        const rhi::ShaderStageMask compute = rhi::ShaderStageBit(rhi::ShaderStage::Compute);
        rhi::BindGroupLayoutEntry entries[5]{};
        entries[0] = {0, rhi::BindingType::SampledTexture, compute, 1, 0, false};
        entries[1] = {1, rhi::BindingType::SampledTexture, compute, 1, 1, false};
        entries[2] = {2, rhi::BindingType::ReadOnlyStorageBuffer, compute, 1, 2, false};
        entries[3] = {3, rhi::BindingType::ReadOnlyStorageBuffer, compute, 1, 3, false};
        entries[4] = {4, rhi::BindingType::StorageBuffer, compute, 1, 0, false};
        rhi::BindGroupLayoutDesc layoutDesc{};
        layoutDesc.entries = Span<const rhi::BindGroupLayoutEntry>(entries, 5);
        auto createdLayout = device.CreateBindGroupLayout(layoutDesc);
        ok = StepR(__LINE__, createdLayout);
        if (ok)
        {
            groupLayout = createdLayout.Value();
            rhi::PipelineLayoutDesc pipelineLayoutDesc{};
            pipelineLayoutDesc.groups[static_cast<u32>(rhi::BindGroupSlot::PerFrame)] = groupLayout;
            auto createdPipelineLayout = device.CreatePipelineLayout(pipelineLayoutDesc);
            ok = StepR(__LINE__, createdPipelineLayout);
            if (ok)
            {
                pipelineLayout = createdPipelineLayout.Value();
                rhi::ComputePipelineDesc pipelineDesc{};
                pipelineDesc.layout = pipelineLayout;
                pipelineDesc.computeShader = shader;
                auto createdPipeline = device.CreateComputePipeline(pipelineDesc);
                if (!createdPipeline)
                {
                    std::printf("pipeline: %s\n", createdPipeline.Error().message);
                }
                ok = createdPipeline.IsOk();
                if (ok)
                {
                    pipeline = createdPipeline.Value();
                }
            }
        }
    }
    if (ok)
    {
        rhi::BindingResource resources[5]{};
        resources[0].type = rhi::BindingType::SampledTexture;
        resources[0].textureView = gpu.PageTableView();
        resources[1].type = rhi::BindingType::SampledTexture;
        resources[1].textureView = gpu.AtlasView();
        resources[2].type = rhi::BindingType::ReadOnlyStorageBuffer;
        resources[2].buffer = queryBuffer;
        resources[3].type = rhi::BindingType::ReadOnlyStorageBuffer;
        resources[3].buffer = paramBuffer;
        resources[4].type = rhi::BindingType::StorageBuffer;
        resources[4].buffer = outBuffer;
        rhi::BindGroupDesc groupDesc{};
        groupDesc.layout = groupLayout;
        groupDesc.resources = Span<const rhi::BindingResource>(resources, 5);
        auto createdGroup = device.CreateBindGroup(groupDesc);
        ok = StepR(__LINE__, createdGroup);
        if (ok)
        {
            group = createdGroup.Value();
        }
    }
    if (ok)
    {
        ok = device
                 .WriteBuffer(queryStage, 0,
                              Span<const Byte>(reinterpret_cast<const Byte*>(queries.data()),
                                               static_cast<usize>(queryBytes)))
                 .IsOk() &&
             device
                 .WriteBuffer(paramStage, 0,
                              Span<const Byte>(reinterpret_cast<const Byte*>(paramWords),
                                               sizeof(paramWords)))
                 .IsOk();
    }
    if (ok)
    {
        rhi::CommandListDesc listDesc{};
        listDesc.queue = rhi::QueueKind::Graphics;
        auto createdList = device.CreateCommandList(listDesc);
        ok = StepR(__LINE__, createdList);
        if (ok)
        {
            rhi::CommandList list = std::move(createdList.Value());
            ok = Step(__LINE__, list.Begin());
            const auto copyIn = [&list](rhi::BufferHandle from, rhi::BufferHandle to,
                                        u64 bytes) -> bool
            {
                rhi::BufferCopy copy{};
                copy.source = from;
                copy.destination = to;
                copy.sizeBytes = bytes;
                return Step(__LINE__, list.CopyBuffer(copy));
            };
            // Staging buffers are copy sources and the device copies are
            // destinations; a buffer already in its state needs no barrier.
            const std::pair<rhi::BufferHandle, rhi::ResourceState> initial[] = {
                {queryStage, rhi::ResourceState::CopySource},
                {paramStage, rhi::ResourceState::CopySource},
                {queryBuffer, rhi::ResourceState::CopyDestination},
                {paramBuffer, rhi::ResourceState::CopyDestination},
                {readback, rhi::ResourceState::CopyDestination}};
            std::vector<rhi::BufferBarrier> initialBarriers;
            for (const auto& [buffer, state] : initial)
            {
                const rhi::ResourceState from = device.BufferStateOf(buffer);
                if (from != state)
                {
                    initialBarriers.push_back({buffer, from, state});
                }
            }
            ok = ok && Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(
                                          initialBarriers.data(), initialBarriers.size())));
            ok = ok && copyIn(queryStage, queryBuffer, queryBytes) &&
                 copyIn(paramStage, paramBuffer, 32);
            const rhi::BufferBarrier toRead[] = {
                {queryBuffer, rhi::ResourceState::CopyDestination, rhi::ResourceState::ShaderRead},
                {paramBuffer, rhi::ResourceState::CopyDestination, rhi::ResourceState::ShaderRead},
                {outBuffer, rhi::ResourceState::Undefined, rhi::ResourceState::ShaderWrite}};
            ok = ok && Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(toRead, 3))) &&
                 Step(__LINE__, list.SetPipeline(pipeline)) &&
                 Step(__LINE__, list.SetBindGroup(rhi::BindGroupSlot::PerFrame, group));
            rhi::DispatchArguments dispatch{};
            dispatch.groupCountX = count / 64u;
            ok = ok && Step(__LINE__, list.Dispatch(dispatch));
            const rhi::BufferBarrier toCopy[] = {
                {outBuffer, rhi::ResourceState::ShaderWrite, rhi::ResourceState::CopySource}};
            ok = ok && Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(toCopy, 1))) &&
                 copyIn(outBuffer, readback, resultBytes) && Step(__LINE__, list.End());
            if (ok)
            {
                const rhi::CommandListHandle handle = list.Handle();
                rhi::SubmitDesc submit{};
                submit.queue = rhi::QueueKind::Graphics;
                submit.commandLists = Span<const rhi::CommandListHandle>(&handle, 1);
                ok = Step(__LINE__, device.Submit(submit)) && Step(__LINE__, device.WaitIdle());
            }
            if (ok)
            {
                results.assign(static_cast<usize>(count) * 4u, 0.0f);
                ok = device
                         .ReadBuffer(readback, 0,
                                     Span<Byte>(reinterpret_cast<Byte*>(results.data()),
                                                static_cast<usize>(resultBytes)))
                         .IsOk();
            }
            (void)device.WaitIdle();
        }
    }
    if (group.IsValid())
    {
        (void)device.DestroyBindGroup(group);
    }
    if (pipeline.IsValid())
    {
        (void)device.DestroyComputePipeline(pipeline);
    }
    if (pipelineLayout.IsValid())
    {
        (void)device.DestroyPipelineLayout(pipelineLayout);
    }
    if (groupLayout.IsValid())
    {
        (void)device.DestroyBindGroupLayout(groupLayout);
    }
    if (shader.IsValid())
    {
        (void)device.DestroyShader(shader);
    }
    for (const rhi::BufferHandle buffer :
         {queryStage, paramStage, queryBuffer, paramBuffer, outBuffer, readback})
    {
        if (buffer.IsValid())
        {
            (void)device.DestroyBuffer(buffer);
        }
    }
    return ok;
}

bool SyncChecked(GpuVirtualTexture& gpu, const VirtualTexture& vt)
{
    const auto status = gpu.Sync(vt);
    if (!status)
    {
        std::printf("sync: %s\n", status.Error().message);
    }
    return status.IsOk();
}

u32 CountMismatches(const VirtualTexture& vt, const std::vector<Query>& queries,
                    const std::vector<f32>& gpuResults, f32& worst)
{
    u32 mismatches = 0;
    worst = 0.0f;
    for (usize i = 0; i < queries.size(); ++i)
    {
        f32 cpu[4];
        vt.Sample(queries[i].u, queries[i].v, queries[i].lod, cpu);
        f32 error = 0.0f;
        for (u32 c = 0; c < 4; ++c)
        {
            error = std::max(error, std::fabs(cpu[c] - gpuResults[i * 4u + c]));
        }
        worst = std::max(worst, error);
        mismatches += error > 2.0e-3f ? 1u : 0u;
    }
    return mismatches;
}

// Routes the RHI's diagnostics to stderr for one test and unhooks them on any
// exit path, including a failed requirement.
struct LogGuard
{
    LogGuard() noexcept
    {
        (void)cyber::core::DefaultLogger().AddSink(&sink);
    }
    ~LogGuard()
    {
        (void)cyber::core::DefaultLogger().RemoveSink(&sink);
    }
    LogGuard(const LogGuard&) = delete;
    LogGuard& operator=(const LogGuard&) = delete;
    cyber::core::ConsoleSink sink;
};

}  // namespace

CG_TEST(gpu_virtual_texture_syncs_only_what_changed)
{
    LogGuard logGuard;
    HeapAllocator allocator;
    rhi::DeviceDesc deviceDesc{};
    deviceDesc.backend = rhi::NullBackend();
    auto createdDevice = rhi::Device::Create(deviceDesc, allocator);
    CG_REQUIRE(createdDevice.IsOk());
    rhi::Device device = std::move(createdDevice.Value());

    TestProducer producer;
    VirtualTexture vt;
    CG_REQUIRE(vt.Initialize(Desc(), producer, allocator).IsOk());
    GpuVirtualTexture gpu;
    CG_REQUIRE(gpu.Initialize(device, vt, GpuVirtualTextureDesc{}, allocator).IsOk());

    // The first sync mirrors every page-table mip and the pinned top page.
    CG_REQUIRE(SyncChecked(gpu, vt));
    CG_EXPECT_EQ(gpu.Stats().pageTableUploads, vt.MipCount());
    CG_EXPECT_TRUE(gpu.Stats().pageUploads >= 1u);
    const u32 firstPages = gpu.Stats().pageUploads;

    // An update with no new requests changes nothing, so a sync uploads nothing.
    (void)vt.Update();
    CG_REQUIRE(SyncChecked(gpu, vt));
    CG_EXPECT_EQ(gpu.Stats().pageUploads, firstPages);
    CG_EXPECT_EQ(gpu.Stats().pageTableUploads, vt.MipCount());

    // Requests load pages; the next syncs upload them and the tables they touch.
    for (u32 frame = 0; frame < 20; ++frame)
    {
        vt.RequestSample(0.31f, 0.62f, 0.0f);
        (void)vt.Update();
        CG_REQUIRE(SyncChecked(gpu, vt));
        std::this_thread::yield();
    }
    vt.Flush();
    CG_REQUIRE(SyncChecked(gpu, vt));
    CG_EXPECT_TRUE(gpu.Stats().pageUploads > firstPages);
    CG_EXPECT_TRUE(gpu.Stats().pageTableUploads > vt.MipCount());
    CG_EXPECT_EQ(gpu.Stats().syncs, 23u);

    gpu.Shutdown();
    vt.Shutdown();
    CG_EXPECT_EQ(device.ReportLeaks(), 0u);
}

CG_TEST(gpu_virtual_texture_shader_matches_cpu_sample_on_d3d12)
{
    LogGuard logGuard;
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
        auto createdDevice = rhi::Device::Create(deviceDesc, allocator);
        if (!createdDevice)
        {
            continue;
        }
        rhi::Device device = std::move(createdDevice.Value());
        ++tested;

        TestProducer producer;
        VirtualTexture vt;
        CG_REQUIRE(vt.Initialize(Desc(), producer, allocator).IsOk());
        GpuVirtualTexture gpu;
        CG_REQUIRE(gpu.Initialize(device, vt, GpuVirtualTextureDesc{}, allocator).IsOk());
        CG_REQUIRE(SyncChecked(gpu, vt));

        const std::vector<Query> queries = MakeQueries(2048, 12345u);
        u32 mismatches = 0;
        u32 fallbacks = 0;
        f32 worst = 0.0f;

        // Round 0: only the pinned top page is resident, so every lookup is an
        // ancestor fallback. Rounds 1 and 2: pages stream in a few per frame,
        // then settle.
        for (u32 round = 0; round < 3; ++round)
        {
            if (round > 0)
            {
                for (u32 frame = 0; frame < (round == 1 ? 6u : 60u); ++frame)
                {
                    for (const Query& query : queries)
                    {
                        vt.RequestSample(query.u, query.v, query.lod);
                    }
                    (void)vt.Update();
                    CG_REQUIRE(SyncChecked(gpu, vt));
                    std::this_thread::yield();
                }
                if (round == 2)
                {
                    vt.Flush();
                    CG_REQUIRE(SyncChecked(gpu, vt));
                }
            }
            for (const Query& query : queries)
            {
                const VirtualPageEntry entry = vt.Resolve(query.u, query.v, query.lod);
                fallbacks += entry.mip > static_cast<u32>(query.lod) ? 1u : 0u;
            }
            std::vector<f32> results;
            CG_REQUIRE(SampleOnGpu(device, gpu, queries, results));
            f32 roundWorst = 0.0f;
            const u32 roundMismatches = CountMismatches(vt, queries, results, roundWorst);
            std::printf("adapter %u round %u: %u of %zu differ, worst %.5f\n", adapterIndex, round,
                        roundMismatches, queries.size(), static_cast<double>(roundWorst));
            mismatches += roundMismatches;
            worst = std::max(worst, roundWorst);
        }
        CG_EXPECT_EQ(mismatches, 0u);
        // Round 0 alone puts nearly every query on a fallback page.
        CG_EXPECT_TRUE(fallbacks > 1000u);
        CG_EXPECT_TRUE(gpu.Stats().pageUploads > 1u);
        gpu.Shutdown();
        vt.Shutdown();
    }
    if (tested == 0)
    {
        CG_SKIP("no Direct3D 12 adapter could be created");
    }
}
