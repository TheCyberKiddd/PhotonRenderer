// GPU-driven virtual texture feedback (Phase 33): the shader that samples a
// virtual texture also counts the pages it asked for, the counters come back
// through a readback ring, and `Resolve` turns them into page requests -- with no
// CPU-side `RequestSample` anywhere. On every Direct3D 12 adapter:
//   1. the counters the shader writes equal, page for page, what
//      `VirtualTexture::RequestSample` would have asked for the same samples;
//   2. a closed loop (sample + feedback on the GPU, read back, stream, upload)
//      brings every sample to the level it wanted and the GPU result matches the
//      CPU sampler.
// The Null backend checks the readback ring's bookkeeping.

#include "test_backend.h"

#include <cyber/core/log.h>
#include <cyber/core/log_sinks.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/renderer/gpu_virtual_texture.h>
#include <cyber/renderer/gpu_virtual_texture_feedback.h>
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
#include <chrono>
#include <cmath>
#include <cstdio>
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
        std::printf("step at line %d failed: %s\n", line, status.Error().message);
    }
    return status.IsOk();
}

struct Query
{
    f32 u, v, lod, unused;
};

struct Random
{
    u32 state;
    f32 Next() noexcept
    {
        state = state * 1664525u + 1013904223u;
        return static_cast<f32>(state >> 8u) * (1.0f / 16777216.0f);
    }
};

std::vector<Query> UniformQueries(u32 count, u32 seed)
{
    Random random{seed};
    std::vector<Query> queries(count);
    for (Query& query : queries)
    {
        query.u = random.Next();
        query.v = random.Next();
        query.lod = random.Next() * 5.5f;
        query.unused = 0.0f;
    }
    return queries;
}

// A view of a small region sampled at four levels: few enough pages for the
// 8x8 cache, spread enough to need every level.
std::vector<Query> RegionQueries(u32 count, u32 seed)
{
    Random random{seed};
    std::vector<Query> queries(count);
    for (u32 i = 0; i < count; ++i)
    {
        queries[i].u = 0.30f + 0.12f * random.Next();
        queries[i].v = 0.55f + 0.12f * random.Next();
        queries[i].lod = static_cast<f32>(i % 4u) + 0.25f;
        queries[i].unused = 0.0f;
    }
    return queries;
}

// The counters the feedback should hold for `queries`: the page
// `RequestSample` picks, in the flat order mip 0 first.
std::vector<u32> ExpectedCounts(const VirtualTexture& vt, const std::vector<Query>& queries)
{
    std::vector<u32> offsets(vt.MipCount());
    u32 total = 0;
    for (u32 mip = 0; mip < vt.MipCount(); ++mip)
    {
        offsets[mip] = total;
        total += vt.PagesX(mip) * vt.PagesY(mip);
    }
    std::vector<u32> counts(total, 0u);
    for (const Query& query : queries)
    {
        const f32 level = std::isfinite(query.lod) ? std::max(query.lod, 0.0f) : 0.0f;
        const u32 mip = std::min(static_cast<u32>(level), vt.MipCount() - 1u);
        const f32 u = query.u - std::floor(query.u);
        const f32 v = query.v - std::floor(query.v);
        const u32 x =
            std::min(static_cast<u32>(u * static_cast<f32>(vt.PagesX(mip))), vt.PagesX(mip) - 1u);
        const u32 y =
            std::min(static_cast<u32>(v * static_cast<f32>(vt.PagesY(mip))), vt.PagesY(mip) - 1u);
        ++counts[offsets[mip] + y * vt.PagesX(mip) + x];
    }
    return counts;
}

const char* const kGlobals = R"hlsl(
Texture2D<uint> g_vtPageTable : register(t0, space0);
Texture2D<float4> g_vtAtlas : register(t1, space0);
RWStructuredBuffer<uint> g_vtFeedback : register(u1, space0);
)hlsl";

const char* const kShader = R"hlsl(
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
    VtFeedback(p, q.xy, q.z);
}
)hlsl";

// One virtual texture on one device with the sampling + feedback shader built
// once; `RunFrame` records, submits and waits for one frame of it.
class Rig
{
public:
    // `fence` belongs to the caller and must outlive every resource the frames use:
    // a resource stays guarded by the fences its submissions signalled.
    Rig(rhi::Device& device, GpuVirtualTexture& gpu, GpuVirtualTextureFeedback& feedback,
        rhi::FenceHandle fence, u32 queryCount) noexcept
        : m_device(device), m_gpu(gpu), m_feedback(feedback), m_count(queryCount), m_fence(fence)
    {
    }

    ~Rig()
    {
        (void)m_device.WaitIdle();
        if (m_group.IsValid())
        {
            (void)m_device.DestroyBindGroup(m_group);
        }
        if (m_pipeline.IsValid())
        {
            (void)m_device.DestroyComputePipeline(m_pipeline);
        }
        if (m_pipelineLayout.IsValid())
        {
            (void)m_device.DestroyPipelineLayout(m_pipelineLayout);
        }
        if (m_groupLayout.IsValid())
        {
            (void)m_device.DestroyBindGroupLayout(m_groupLayout);
        }
        if (m_shader.IsValid())
        {
            (void)m_device.DestroyShader(m_shader);
        }
        for (const rhi::BufferHandle buffer :
             {m_queryStage, m_paramStage, m_queryBuffer, m_paramBuffer, m_outBuffer, m_readback})
        {
            if (buffer.IsValid())
            {
                (void)m_device.DestroyBuffer(buffer);
            }
        }
    }

    [[nodiscard]] bool Build() noexcept
    {
        const u64 queryBytes = static_cast<u64>(m_count) * sizeof(Query);
        const u64 resultBytes = static_cast<u64>(m_count) * 16u;
        m_queryStage =
            Make(rhi::BufferUsage::HostVisible | rhi::BufferUsage::CopySource, queryBytes, 0);
        m_paramStage = Make(rhi::BufferUsage::HostVisible | rhi::BufferUsage::CopySource, 32, 0);
        m_queryBuffer =
            Make(rhi::BufferUsage::Storage | rhi::BufferUsage::CopyDestination, queryBytes, 16);
        m_paramBuffer = Make(rhi::BufferUsage::Storage | rhi::BufferUsage::CopyDestination, 32, 4);
        m_outBuffer =
            Make(rhi::BufferUsage::Storage | rhi::BufferUsage::CopySource, resultBytes, 16);
        m_readback =
            Make(rhi::BufferUsage::Readback | rhi::BufferUsage::CopyDestination, resultBytes, 0);
        if (!m_queryStage.IsValid() || !m_paramStage.IsValid() || !m_queryBuffer.IsValid() ||
            !m_paramBuffer.IsValid() || !m_outBuffer.IsValid() || !m_readback.IsValid())
        {
            return false;
        }

        std::string source = kGlobals;
        source += GpuVirtualTexture::ShaderSource();
        source += GpuVirtualTextureFeedback::ShaderSource();
        source += kShader;
        rhi::ShaderDesc shaderDesc{};
        shaderDesc.stage = rhi::ShaderStage::Compute;
        shaderDesc.format = rhi::ShaderBytecodeFormat::HlslSource;
        shaderDesc.code =
            Span<const Byte>(reinterpret_cast<const Byte*>(source.data()), source.size());
        shaderDesc.entryPoint = "main";
        shaderDesc.label = "vt_feedback_test";
        auto shader = m_device.CreateShader(shaderDesc);
        if (!shader)
        {
            std::printf("shader: %s\n", shader.Error().message);
            return false;
        }
        m_shader = shader.Value();

        const rhi::ShaderStageMask compute = rhi::ShaderStageBit(rhi::ShaderStage::Compute);
        rhi::BindGroupLayoutEntry entries[6]{};
        entries[0] = {0, rhi::BindingType::SampledTexture, compute, 1, 0, false};
        entries[1] = {1, rhi::BindingType::SampledTexture, compute, 1, 1, false};
        entries[2] = {2, rhi::BindingType::ReadOnlyStorageBuffer, compute, 1, 2, false};
        entries[3] = {3, rhi::BindingType::ReadOnlyStorageBuffer, compute, 1, 3, false};
        entries[4] = {4, rhi::BindingType::StorageBuffer, compute, 1, 0, false};
        entries[5] = {5, rhi::BindingType::StorageBuffer, compute, 1, 1, false};
        rhi::BindGroupLayoutDesc layoutDesc{};
        layoutDesc.entries = Span<const rhi::BindGroupLayoutEntry>(entries, 6);
        auto layout = m_device.CreateBindGroupLayout(layoutDesc);
        if (!layout)
        {
            return false;
        }
        m_groupLayout = layout.Value();
        rhi::PipelineLayoutDesc pipelineLayoutDesc{};
        pipelineLayoutDesc.groups[static_cast<u32>(rhi::BindGroupSlot::PerFrame)] = m_groupLayout;
        auto pipelineLayout = m_device.CreatePipelineLayout(pipelineLayoutDesc);
        if (!pipelineLayout)
        {
            return false;
        }
        m_pipelineLayout = pipelineLayout.Value();
        rhi::ComputePipelineDesc pipelineDesc{};
        pipelineDesc.layout = m_pipelineLayout;
        pipelineDesc.computeShader = m_shader;
        auto pipeline = m_device.CreateComputePipeline(pipelineDesc);
        if (!pipeline)
        {
            std::printf("pipeline: %s\n", pipeline.Error().message);
            return false;
        }
        m_pipeline = pipeline.Value();

        rhi::BindingResource resources[6]{};
        resources[0].type = rhi::BindingType::SampledTexture;
        resources[0].textureView = m_gpu.PageTableView();
        resources[1].type = rhi::BindingType::SampledTexture;
        resources[1].textureView = m_gpu.AtlasView();
        resources[2].type = rhi::BindingType::ReadOnlyStorageBuffer;
        resources[2].buffer = m_queryBuffer;
        resources[3].type = rhi::BindingType::ReadOnlyStorageBuffer;
        resources[3].buffer = m_paramBuffer;
        resources[4].type = rhi::BindingType::StorageBuffer;
        resources[4].buffer = m_outBuffer;
        resources[5].type = rhi::BindingType::StorageBuffer;
        resources[5].buffer = m_feedback.Buffer();
        rhi::BindGroupDesc groupDesc{};
        groupDesc.layout = m_groupLayout;
        groupDesc.resources = Span<const rhi::BindingResource>(resources, 6);
        auto group = m_device.CreateBindGroup(groupDesc);
        if (!group)
        {
            return false;
        }
        m_group = group.Value();

        const GpuVirtualTextureParams params = m_gpu.Params();
        const u32 words[8] = {params.width,
                              params.height,
                              params.pageSize,
                              params.border,
                              params.stride,
                              params.mipCount,
                              0u,
                              0u};
        return m_device
            .WriteBuffer(m_paramStage, 0,
                         Span<const Byte>(reinterpret_cast<const Byte*>(words), sizeof(words)))
            .IsOk();
    }

    // Samples `queries` and records their feedback in one submission, waits for
    // it and returns the shader's results.
    [[nodiscard]] bool RunFrame(const std::vector<Query>& queries,
                                std::vector<f32>& results) noexcept
    {
        const u64 queryBytes = static_cast<u64>(m_count) * sizeof(Query);
        const u64 resultBytes = static_cast<u64>(m_count) * 16u;
        if (!Step(__LINE__, m_device.WriteBuffer(
                                m_queryStage, 0,
                                Span<const Byte>(reinterpret_cast<const Byte*>(queries.data()),
                                                 static_cast<usize>(queryBytes)))))
        {
            return false;
        }
        rhi::CommandListDesc listDesc{};
        listDesc.queue = rhi::QueueKind::Graphics;
        auto created = m_device.CreateCommandList(listDesc);
        if (!created)
        {
            return false;
        }
        rhi::CommandList list = std::move(created.Value());
        bool ok = Step(__LINE__, list.Begin());

        // Everything the frame copies goes through one barrier batch, built from
        // the states the device tracks.
        const std::pair<rhi::BufferHandle, rhi::ResourceState> toCopy[] = {
            {m_queryStage, rhi::ResourceState::CopySource},
            {m_paramStage, rhi::ResourceState::CopySource},
            {m_queryBuffer, rhi::ResourceState::CopyDestination},
            {m_paramBuffer, rhi::ResourceState::CopyDestination},
            {m_readback, rhi::ResourceState::CopyDestination}};
        std::vector<rhi::BufferBarrier> barriers;
        for (const auto& [buffer, state] : toCopy)
        {
            const rhi::ResourceState from = m_device.BufferStateOf(buffer);
            if (from != state)
            {
                barriers.push_back({buffer, from, state});
            }
        }
        ok = ok && Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(barriers.data(),
                                                                              barriers.size())));
        rhi::BufferCopy copyQueries{};
        copyQueries.source = m_queryStage;
        copyQueries.destination = m_queryBuffer;
        copyQueries.sizeBytes = queryBytes;
        rhi::BufferCopy copyParams{};
        copyParams.source = m_paramStage;
        copyParams.destination = m_paramBuffer;
        copyParams.sizeBytes = 32;
        ok = ok && Step(__LINE__, list.CopyBuffer(copyQueries)) &&
             Step(__LINE__, list.CopyBuffer(copyParams));
        const rhi::BufferBarrier toRead[] = {
            {m_queryBuffer, rhi::ResourceState::CopyDestination, rhi::ResourceState::ShaderRead},
            {m_paramBuffer, rhi::ResourceState::CopyDestination, rhi::ResourceState::ShaderRead},
            {m_outBuffer, m_device.BufferStateOf(m_outBuffer), rhi::ResourceState::ShaderWrite}};
        ok = ok && Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(toRead, 3)));
        // The feedback counters are cleared, written by the dispatch below and
        // copied to the readback ring: Begin ... End.
        ok = ok && Step(__LINE__, m_feedback.Begin(list));
        ok = ok && Step(__LINE__, list.SetPipeline(m_pipeline)) &&
             Step(__LINE__, list.SetBindGroup(rhi::BindGroupSlot::PerFrame, m_group));
        rhi::DispatchArguments dispatch{};
        dispatch.groupCountX = m_count / 64u;
        ok = ok && Step(__LINE__, list.Dispatch(dispatch));
        const rhi::BufferBarrier toCopyOut[] = {
            {m_outBuffer, rhi::ResourceState::ShaderWrite, rhi::ResourceState::CopySource}};
        ok = ok && Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(toCopyOut, 1)));
        rhi::BufferCopy copyOut{};
        copyOut.source = m_outBuffer;
        copyOut.destination = m_readback;
        copyOut.sizeBytes = resultBytes;
        ok = ok && Step(__LINE__, list.CopyBuffer(copyOut));
        const u64 value = ++m_value;
        ok = ok && Step(__LINE__, m_feedback.End(list, m_fence, value));
        ok = ok && Step(__LINE__, list.End());
        if (!ok)
        {
            return false;
        }
        const rhi::CommandListHandle handle = list.Handle();
        const rhi::FenceSignal signal{m_fence, value};
        rhi::SubmitDesc submit{};
        submit.queue = rhi::QueueKind::Graphics;
        submit.commandLists = Span<const rhi::CommandListHandle>(&handle, 1);
        submit.signals = Span<const rhi::FenceSignal>(&signal, 1);
        if (!Step(__LINE__, m_device.Submit(submit)) ||
            !Step(__LINE__, m_device.WaitFence(m_fence, value)))
        {
            return false;
        }
        results.assign(static_cast<usize>(m_count) * 4u, 0.0f);
        return Step(__LINE__,
                    m_device.ReadBuffer(m_readback, 0,
                                        Span<Byte>(reinterpret_cast<Byte*>(results.data()),
                                                   static_cast<usize>(resultBytes))));
    }

private:
    [[nodiscard]] rhi::BufferHandle Make(rhi::BufferUsage usage, u64 bytes, u32 stride) noexcept
    {
        rhi::BufferDesc desc{};
        desc.usage = usage;
        desc.sizeBytes = bytes;
        desc.elementStride = stride;
        desc.label = "vt_feedback_test";
        auto created = m_device.CreateBuffer(desc);
        return created ? created.Value() : rhi::BufferHandle{};
    }

    rhi::Device& m_device;
    GpuVirtualTexture& m_gpu;
    GpuVirtualTextureFeedback& m_feedback;
    u32 m_count;
    u64 m_value = 0;
    rhi::FenceHandle m_fence{};
    rhi::BufferHandle m_queryStage{};
    rhi::BufferHandle m_paramStage{};
    rhi::BufferHandle m_queryBuffer{};
    rhi::BufferHandle m_paramBuffer{};
    rhi::BufferHandle m_outBuffer{};
    rhi::BufferHandle m_readback{};
    rhi::ShaderHandle m_shader{};
    rhi::BindGroupLayoutHandle m_groupLayout{};
    rhi::PipelineLayoutHandle m_pipelineLayout{};
    rhi::PipelineHandle m_pipeline{};
    rhi::BindGroupHandle m_group{};
};

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

bool AllSharp(const VirtualTexture& vt, const std::vector<Query>& queries)
{
    for (const Query& query : queries)
    {
        if (vt.Resolve(query.u, query.v, query.lod).mip > static_cast<u32>(query.lod))
        {
            return false;
        }
    }
    return true;
}

}  // namespace

CG_TEST(gpu_feedback_ring_tracks_frames_and_stalls_when_full)
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
    GpuVirtualTextureFeedback feedback;
    GpuVirtualTextureFeedbackDesc feedbackDesc;
    feedbackDesc.framesInFlight = 2;
    CG_REQUIRE(feedback.Initialize(device, vt, feedbackDesc, allocator).IsOk());
    u32 pages = 0;
    for (u32 mip = 0; mip < vt.MipCount(); ++mip)
    {
        pages += vt.PagesX(mip) * vt.PagesY(mip);
    }
    CG_EXPECT_EQ(feedback.TotalPages(), pages);

    auto fence = device.CreateFence(rhi::FenceDesc{});
    CG_REQUIRE(fence.IsOk());
    const auto recordFrame = [&](u64 value)
    {
        rhi::CommandListDesc listDesc{};
        listDesc.queue = rhi::QueueKind::Graphics;
        auto created = device.CreateCommandList(listDesc);
        CG_REQUIRE(created.IsOk());
        rhi::CommandList list = std::move(created.Value());
        CG_REQUIRE(Step(__LINE__, list.Begin()));
        CG_REQUIRE(Step(__LINE__, feedback.Begin(list)));
        CG_REQUIRE(Step(__LINE__, feedback.End(list, fence.Value(), value)));
        CG_REQUIRE(Step(__LINE__, list.End()));
    };

    // Nothing has finished, so nothing is read back.
    recordFrame(1);
    CG_EXPECT_EQ(feedback.Resolve(), 0u);
    CG_EXPECT_EQ(feedback.Stats().resolvedFrames, 0u);

    // The fence reaches the frame: it is read back (the Null backend executes no
    // work, so what the counters hold is its own business).
    CG_REQUIRE(device.SignalFence(fence.Value(), 1).IsOk());
    (void)feedback.Resolve();
    CG_EXPECT_EQ(feedback.Stats().resolvedFrames, 1u);
    CG_EXPECT_EQ(feedback.LastCounts().Size(), static_cast<usize>(pages));

    // Two slots: a third frame without a resolve in between waits for the oldest.
    recordFrame(2);
    recordFrame(3);
    CG_REQUIRE(device.SignalFence(fence.Value(), 3).IsOk());
    recordFrame(4);
    CG_EXPECT_EQ(feedback.Stats().stalls, 1u);
    CG_REQUIRE(device.SignalFence(fence.Value(), 4).IsOk());
    (void)feedback.Resolve();
    CG_EXPECT_EQ(feedback.Stats().resolvedFrames, 4u);

    feedback.Shutdown();
    vt.Shutdown();
    // The fence goes last: resources stay guarded by the fences of the
    // submissions that used them.
    CG_REQUIRE(device.DestroyFence(fence.Value()).IsOk());
    CG_EXPECT_EQ(device.ReportLeaks(), 0u);
}

CG_TEST(gpu_feedback_counts_and_closed_loop_match_cpu_on_d3d12)
{
    LogGuard logGuard;
    const rhi::Backend* backend = cyber::rhi::test::TestBackend();
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
        auto createdFence = device.CreateFence(rhi::FenceDesc{});
        CG_REQUIRE(createdFence.IsOk());
        const rhi::FenceHandle fence = createdFence.Value();

        // --- 1. The counters equal what the CPU would have requested. -------
        {
            TestProducer producer;
            VirtualTexture vt;
            CG_REQUIRE(vt.Initialize(Desc(), producer, allocator).IsOk());
            GpuVirtualTexture gpu;
            CG_REQUIRE(gpu.Initialize(device, vt, GpuVirtualTextureDesc{}, allocator).IsOk());
            CG_REQUIRE(Step(__LINE__, gpu.Sync(vt)));
            GpuVirtualTextureFeedback feedback;
            CG_REQUIRE(
                feedback.Initialize(device, vt, GpuVirtualTextureFeedbackDesc{}, allocator).IsOk());
            const std::vector<Query> queries = UniformQueries(2048, 777u);
            {
                Rig rig(device, gpu, feedback, fence, 2048);
                CG_REQUIRE(rig.Build());
                std::vector<f32> results;
                CG_REQUIRE(rig.RunFrame(queries, results));
                const u32 requested = feedback.Resolve();
                CG_REQUIRE(feedback.Stats().resolvedFrames == 1u);

                const std::vector<u32> expected = ExpectedCounts(vt, queries);
                CG_REQUIRE(feedback.LastCounts().Size() == expected.size());
                u32 wrong = 0;
                u32 distinct = 0;
                u64 total = 0;
                for (usize i = 0; i < expected.size(); ++i)
                {
                    wrong += feedback.LastCounts()[i] != expected[i] ? 1u : 0u;
                    distinct += expected[i] != 0 ? 1u : 0u;
                    total += feedback.LastCounts()[i];
                }
                std::printf("adapter %u counters: %u of %zu pages differ, %llu samples, %u "
                            "distinct pages requested\n",
                            adapterIndex, wrong, expected.size(),
                            static_cast<unsigned long long>(total), requested);
                CG_EXPECT_EQ(wrong, 0u);
                CG_EXPECT_EQ(total, 2048ull);
                CG_EXPECT_EQ(requested, distinct);
                // The requests reached the texture: its next update sees exactly
                // those pages, though no CPU sampler asked for any.
                const VirtualTextureStats stats = vt.Update();
                CG_EXPECT_EQ(stats.requestedThisFrame, distinct);
            }
            feedback.Shutdown();
            gpu.Shutdown();
            vt.Shutdown();
        }

        // --- 2. Closed loop: the GPU's own feedback streams what it samples. -
        {
            TestProducer producer;
            VirtualTexture vt;
            CG_REQUIRE(vt.Initialize(Desc(), producer, allocator).IsOk());
            GpuVirtualTexture gpu;
            CG_REQUIRE(gpu.Initialize(device, vt, GpuVirtualTextureDesc{}, allocator).IsOk());
            CG_REQUIRE(Step(__LINE__, gpu.Sync(vt)));
            GpuVirtualTextureFeedback feedback;
            CG_REQUIRE(
                feedback.Initialize(device, vt, GpuVirtualTextureFeedbackDesc{}, allocator).IsOk());
            const std::vector<Query> queries = RegionQueries(256, 4242u);
            {
                Rig rig(device, gpu, feedback, fence, 256);
                CG_REQUIRE(rig.Build());
                std::vector<f32> results;
                const bool sharpAtStart = AllSharp(vt, queries);
                u32 framesToSharp = 0;
                for (u32 frame = 0; frame < 600 && !AllSharp(vt, queries); ++frame)
                {
                    CG_REQUIRE(rig.RunFrame(queries, results));
                    (void)feedback.Resolve();
                    (void)vt.Update();
                    CG_REQUIRE(Step(__LINE__, gpu.Sync(vt)));
                    ++framesToSharp;
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
                CG_REQUIRE(rig.RunFrame(queries, results));
                f32 worst = 0.0f;
                const u32 mismatches = CountMismatches(vt, queries, results, worst);
                std::printf("adapter %u closed loop: sharp after %u frames (%u feedback frames "
                            "resolved), %u of %zu differ from the CPU sampler, worst %.5f\n",
                            adapterIndex, framesToSharp, feedback.Stats().resolvedFrames,
                            mismatches, queries.size(), static_cast<double>(worst));
                CG_EXPECT_TRUE(!sharpAtStart);
                CG_EXPECT_TRUE(AllSharp(vt, queries));
                CG_EXPECT_EQ(mismatches, 0u);
                CG_EXPECT_TRUE(feedback.Stats().resolvedFrames > 0u);
            }
            feedback.Shutdown();
            gpu.Shutdown();
            vt.Shutdown();
        }
        CG_REQUIRE(device.DestroyFence(fence).IsOk());
        CG_EXPECT_EQ(device.ReportLeaks(), 0u);
    }
    if (tested == 0)
    {
        CG_SKIP("no Direct3D 12 adapter could be created");
    }
}
