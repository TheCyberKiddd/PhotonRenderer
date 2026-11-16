// GPU-rendered texture streaming (Phase 33): six reserved textures are streamed
// through a tile heap far smaller than their full chains while a compute
// shader, every frame, samples each of them at the level the frame wants
// (clamped to what is resident) and the results are checked texel for texel
// against the pattern the sources generate. The shader never reads an unmapped
// tile, the evictions and refinements that happen between frames never corrupt
// a sample, and the streaming work stays inside a frame budget.

#include "test_backend.h"

#include <cyber/core/log.h>
#include <cyber/core/log_sinks.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/renderer/gpu_texture_pool.h>
#include <cyber/renderer/texture_streaming.h>
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
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using cyber::core::Byte;
using cyber::core::f32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::usize;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using namespace cyber::renderer;
namespace rhi = cyber::rhi;

constexpr u32 kTextures = 6;
constexpr u32 kSize = 1024;
constexpr u32 kSamples = 64;

// Every byte is a function of (texture, level, offset), so any texel the shader
// returns can be checked without storing the image.
Byte PatternByte(u32 id, u32 level, usize offset) noexcept
{
    return static_cast<Byte>((offset * 131u + level * 29u + id * 7u + (offset >> 9u)) & 0xFFu);
}

class PatternSource final : public TextureMipSource
{
public:
    explicit PatternSource(u32 id) noexcept : m_id(id)
    {
        for (u32 size = kSize; size >= 1; size /= 2)
        {
            ++m_levels;
        }
    }
    u32 LevelCount() const noexcept override
    {
        return m_levels;
    }
    StreamingLevelInfo Level(u32 level) const noexcept override
    {
        const u32 size = std::max(kSize >> level, 1u);
        return StreamingLevelInfo{size, size, static_cast<u64>(size) * size * 4u};
    }
    bool ReadLevel(u32 level, Span<Byte> destination) noexcept override
    {
        for (usize i = 0; i < destination.Size(); ++i)
        {
            destination[i] = PatternByte(m_id, level, i);
        }
        return true;
    }
    [[nodiscard]] rhi::TextureDesc Desc() const noexcept
    {
        rhi::TextureDesc desc{};
        desc.format = rhi::Format::RGBA8Unorm;
        desc.usage = rhi::TextureUsage::Sampled | rhi::TextureUsage::CopySource;
        desc.width = kSize;
        desc.height = kSize;
        desc.mipLevels = m_levels;
        return desc;
    }

private:
    u32 m_id = 0;
    u32 m_levels = 0;
};

template <class S>
bool Step(int line, const S& status)
{
    if (!status)
    {
        std::printf("gpu step at line %d failed: %s\n", line, status.Error().message);
    }
    return status.IsOk();
}

// Six separate bindings rather than one `Texture2D Tex[6]` array: on WARP, indexing
// a bound array of texture views returned element 0 for every index (the
// hardware adapter is correct), so this test stays on the portable form. See
// the Phase 33 report, known issues.
const char* const kShader = R"hlsl(
Texture2D<float4> Tex0 : register(t0, space0);
Texture2D<float4> Tex1 : register(t1, space0);
Texture2D<float4> Tex2 : register(t2, space0);
Texture2D<float4> Tex3 : register(t3, space0);
Texture2D<float4> Tex4 : register(t4, space0);
Texture2D<float4> Tex5 : register(t5, space0);
StructuredBuffer<uint> Params : register(t6, space0);
RWStructuredBuffer<uint> Output : register(u0, space0);

uint Pack(float4 c)
{
    uint4 b = (uint4)round(saturate(c) * 255.0f);
    return b.x | (b.y << 8u) | (b.z << 16u) | (b.w << 24u);
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    [unroll]
    for (uint t = 0; t < 6; ++t)
    {
        uint relMip = Params[t * 2u];
        uint dim = Params[t * 2u + 1u];
        uint x = (id.x * 37u + t * 11u) % dim;
        uint y = (id.x * 53u + t * 5u) % dim;
        int3 c = int3((int)x, (int)y, (int)relMip);
        float4 v = t == 0 ? Tex0.Load(c) : t == 1 ? Tex1.Load(c) : t == 2 ? Tex2.Load(c) : t == 3 ? Tex3.Load(c) : t == 4 ? Tex4.Load(c) : Tex5.Load(c);
        Output[t * 64u + id.x] = Pack(v);
    }
}
)hlsl";

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

CG_TEST(gpu_streamed_textures_render_correctly_under_a_small_budget_on_d3d12)
{
    LogGuard logGuard;
    // Progress lines must survive an abnormal exit of a driver or debug layer.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
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
        // Full validation, GPU-based included. A reserved texture used to end the
        // process under the debug layer when it was released with a tile-mapping
        // update still queued; the backend now waits for those updates first.
        deviceDesc.validation = rhi::ValidationLevel::Exhaustive;
        auto createdDevice = rhi::Device::Create(deviceDesc, allocator);
        if (!createdDevice ||
            !createdDevice.Value().Capabilities().features.Has(rhi::Feature::ReservedTextures))
        {
            continue;
        }
        rhi::Device device = std::move(createdDevice.Value());
        ++tested;

        std::vector<std::unique_ptr<PatternSource>> sources;
        for (u32 i = 0; i < kTextures; ++i)
        {
            sources.push_back(std::make_unique<PatternSource>(i));
        }
        // Budget: a quarter of everything the textures would need in full, so
        // at most one or two can be sharp at a time.
        u32 totalTiles = 0;
        {
            rhi::TextureDesc probe = sources[0]->Desc();
            probe.reserved = true;
            probe.usage = probe.usage | rhi::TextureUsage::CopyDestination;
            auto texture = device.CreateTexture(probe);
            CG_REQUIRE(texture.IsOk());
            totalTiles = device.TextureTilingOf(texture.Value()).totalTileCount * kTextures;
            (void)device.DestroyTexture(texture.Value());
        }
        const u32 budgetTiles = totalTiles / 4u;

        TextureStreamerDesc streamerDesc;
        streamerDesc.budgetBytes = static_cast<u64>(budgetTiles) * rhi::kTileSizeBytes;
        streamerDesc.pageBytes = static_cast<u32>(rhi::kTileSizeBytes);
        streamerDesc.ioThreads = 2;
        streamerDesc.maxInFlight = 8;
        streamerDesc.maxUploadBytesPerFrame = 2ull << 20u;
        streamerDesc.unusedFrameLimit = 4;
        TextureStreamer streamer;
        CG_REQUIRE(streamer.Initialize(streamerDesc, allocator).IsOk());
        GpuTexturePool pool;
        GpuTexturePoolDesc poolDesc;
        poolDesc.tileCount = budgetTiles;
        CG_REQUIRE(pool.Initialize(device, poolDesc, allocator).IsOk());
        std::vector<TextureHandle> handles;
        for (auto& source : sources)
        {
            auto added = pool.AddTexture(streamer, *source, source->Desc(), "streamed");
            CG_REQUIRE(added.IsOk());
            handles.push_back(added.Value());
        }

        // Pipeline and buffers, built once.
        std::string hlsl = kShader;
        rhi::ShaderDesc shaderDesc{};
        shaderDesc.stage = rhi::ShaderStage::Compute;
        shaderDesc.format = rhi::ShaderBytecodeFormat::HlslSource;
        shaderDesc.code = Span<const Byte>(reinterpret_cast<const Byte*>(hlsl.data()), hlsl.size());
        shaderDesc.entryPoint = "main";
        shaderDesc.label = "streaming_render_test";
        auto shader = device.CreateShader(shaderDesc);
        CG_REQUIRE(shader.IsOk());
        const rhi::ShaderStageMask compute = rhi::ShaderStageBit(rhi::ShaderStage::Compute);
        rhi::BindGroupLayoutEntry entries[kTextures + 2]{};
        for (u32 i = 0; i < kTextures; ++i)
        {
            entries[i] = {i, rhi::BindingType::SampledTexture, compute, 1, i, false};
        }
        entries[kTextures] = {kTextures, rhi::BindingType::ReadOnlyStorageBuffer, compute, 1, 6,
                              false};
        entries[kTextures + 1] = {kTextures + 1, rhi::BindingType::StorageBuffer, compute, 1, 0,
                                  false};
        rhi::BindGroupLayoutDesc layoutDesc{};
        layoutDesc.entries = Span<const rhi::BindGroupLayoutEntry>(entries, kTextures + 2);
        auto groupLayout = device.CreateBindGroupLayout(layoutDesc);
        CG_REQUIRE(groupLayout.IsOk());
        rhi::PipelineLayoutDesc pipelineLayoutDesc{};
        pipelineLayoutDesc.groups[static_cast<u32>(rhi::BindGroupSlot::PerFrame)] =
            groupLayout.Value();
        auto pipelineLayout = device.CreatePipelineLayout(pipelineLayoutDesc);
        CG_REQUIRE(pipelineLayout.IsOk());
        rhi::ComputePipelineDesc pipelineDesc{};
        pipelineDesc.layout = pipelineLayout.Value();
        pipelineDesc.computeShader = shader.Value();
        auto pipeline = device.CreateComputePipeline(pipelineDesc);
        CG_REQUIRE(pipeline.IsOk());

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
        constexpr u64 kParamBytes = kTextures * 2u * sizeof(u32);
        constexpr u64 kResultBytes = static_cast<u64>(kTextures) * kSamples * sizeof(u32);
        const rhi::BufferHandle paramStage =
            makeBuffer(rhi::BufferUsage::HostVisible | rhi::BufferUsage::CopySource, kParamBytes, 0,
                       "param_stage");
        const rhi::BufferHandle paramBuffer =
            makeBuffer(rhi::BufferUsage::Storage | rhi::BufferUsage::CopyDestination, kParamBytes,
                       4, "params");
        const rhi::BufferHandle outBuffer = makeBuffer(
            rhi::BufferUsage::Storage | rhi::BufferUsage::CopySource, kResultBytes, 4, "results");
        const rhi::BufferHandle readback =
            makeBuffer(rhi::BufferUsage::Readback | rhi::BufferUsage::CopyDestination, kResultBytes,
                       0, "readback");
        CG_REQUIRE(paramStage.IsValid() && paramBuffer.IsValid() && outBuffer.IsValid() &&
                   readback.IsValid());

        // Load every tail before the first frame, so each texture has a level
        // to sample.
        const auto frame = [&](const std::vector<f32>& coverage, const std::vector<f32>& wanted)
        {
            CG_REQUIRE(Step(__LINE__, pool.BeginFrame()));
            for (u32 i = 0; i < kTextures; ++i)
            {
                streamer.ReportUsage(handles[i], wanted[i], coverage[i]);
            }
            (void)streamer.Update(pool);
            CG_REQUIRE(Step(__LINE__, pool.EndFrame()));
        };
        const std::vector<f32> startWanted(kTextures, 0.0f);
        const std::vector<f32> startCoverage(kTextures, 0.1f);
        for (u32 warm = 0; warm < 400; ++warm)
        {
            frame(startCoverage, startWanted);
            bool ready = true;
            for (const TextureHandle handle : handles)
            {
                ready = ready && streamer.ResidentLevel(handle) < sources[0]->LevelCount();
            }
            if (ready)
            {
                break;
            }
            std::this_thread::yield();
        }

        u64 mismatches = 0;
        u64 samplesChecked = 0;
        u32 sharpest = sources[0]->LevelCount();
        double worstStreamMs = 0.0;
        std::vector<double> streamMs;
        constexpr u32 kFrames = 240;
        for (u32 f = 0; f < kFrames; ++f)
        {
            // Four phases of changing interest: one texture at a time wants to
            // be sharp, and the rest fall back to coarse levels.
            const u32 phase = (f / 30u) % kTextures;
            std::vector<f32> coverage(kTextures);
            std::vector<f32> wanted(kTextures);
            for (u32 i = 0; i < kTextures; ++i)
            {
                const bool focus = i == phase || i == (phase + 1u) % kTextures;
                coverage[i] = focus ? 0.5f : 0.02f;
                wanted[i] = focus ? 0.0f : 4.0f;
            }
            const auto begun = std::chrono::steady_clock::now();
            frame(coverage, wanted);
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begun)
                    .count();
            streamMs.push_back(ms);
            worstStreamMs = std::max(worstStreamMs, ms);
            // Pace the frames like a real application: streaming IO is wall-clock
            // work, and a loop that never waits starves it, most of all in Debug.
            std::this_thread::sleep_for(std::chrono::milliseconds(8));

            // The level each texture is sampled at: what was asked for, clamped
            // to what is resident.
            u32 absoluteMip[kTextures];
            rhi::TextureViewHandle views[kTextures];
            u32 paramWords[kTextures * 2];
            for (u32 i = 0; i < kTextures; ++i)
            {
                const u32 levels = sources[i]->LevelCount();
                const u32 resident = streamer.ResidentLevel(handles[i]);
                sharpest = std::min(sharpest, resident);
                absoluteMip[i] = std::max(static_cast<u32>(wanted[i]), resident);
                absoluteMip[i] = std::min(absoluteMip[i], levels - 1u);
                paramWords[i * 2u] = absoluteMip[i] - resident;
                paramWords[i * 2u + 1u] = std::max(kSize >> absoluteMip[i], 1u);
                rhi::TextureViewDesc viewDesc{};
                viewDesc.texture = pool.Texture(handles[i]);
                viewDesc.format = rhi::Format::RGBA8Unorm;
                viewDesc.range.baseMipLevel = resident;
                viewDesc.range.mipLevelCount = levels - resident;
                viewDesc.label = "streamed_view";
                auto view = device.CreateTextureView(viewDesc);
                CG_REQUIRE(view.IsOk());
                views[i] = view.Value();
            }
            rhi::BindingResource resources[kTextures + 2]{};
            for (u32 i = 0; i < kTextures; ++i)
            {
                resources[i].type = rhi::BindingType::SampledTexture;
                resources[i].textureView = views[i];
            }
            resources[kTextures].type = rhi::BindingType::ReadOnlyStorageBuffer;
            resources[kTextures].buffer = paramBuffer;
            resources[kTextures + 1].type = rhi::BindingType::StorageBuffer;
            resources[kTextures + 1].buffer = outBuffer;
            rhi::BindGroupDesc groupDesc{};
            groupDesc.layout = groupLayout.Value();
            groupDesc.resources = Span<const rhi::BindingResource>(resources, kTextures + 2);
            auto group = device.CreateBindGroup(groupDesc);
            CG_REQUIRE(group.IsOk());

            CG_REQUIRE(
                Step(__LINE__,
                     device.WriteBuffer(paramStage, 0,
                                        Span<const Byte>(reinterpret_cast<const Byte*>(paramWords),
                                                         sizeof(paramWords)))));
            rhi::CommandListDesc listDesc{};
            listDesc.queue = rhi::QueueKind::Graphics;
            auto createdList = device.CreateCommandList(listDesc);
            CG_REQUIRE(createdList.IsOk());
            rhi::CommandList list = std::move(createdList.Value());
            CG_REQUIRE(Step(__LINE__, list.Begin()));
            const std::pair<rhi::BufferHandle, rhi::ResourceState> toCopy[] = {
                {paramStage, rhi::ResourceState::CopySource},
                {paramBuffer, rhi::ResourceState::CopyDestination},
                {readback, rhi::ResourceState::CopyDestination}};
            std::vector<rhi::BufferBarrier> barriers;
            for (const auto& [buffer, state] : toCopy)
            {
                const rhi::ResourceState from = device.BufferStateOf(buffer);
                if (from != state)
                {
                    barriers.push_back({buffer, from, state});
                }
            }
            CG_REQUIRE(Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(
                                          barriers.data(), barriers.size()))));
            rhi::BufferCopy copy{};
            copy.source = paramStage;
            copy.destination = paramBuffer;
            copy.sizeBytes = kParamBytes;
            CG_REQUIRE(Step(__LINE__, list.CopyBuffer(copy)));
            const rhi::BufferBarrier toShader[] = {
                {paramBuffer, rhi::ResourceState::CopyDestination, rhi::ResourceState::ShaderRead},
                {outBuffer, device.BufferStateOf(outBuffer), rhi::ResourceState::ShaderWrite}};
            CG_REQUIRE(Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(toShader, 2))));
            CG_REQUIRE(Step(__LINE__, list.SetPipeline(pipeline.Value())));
            CG_REQUIRE(
                Step(__LINE__, list.SetBindGroup(rhi::BindGroupSlot::PerFrame, group.Value())));
            rhi::DispatchArguments dispatch{};
            dispatch.groupCountX = 1;
            CG_REQUIRE(Step(__LINE__, list.Dispatch(dispatch)));
            const rhi::BufferBarrier toRead[] = {
                {outBuffer, rhi::ResourceState::ShaderWrite, rhi::ResourceState::CopySource}};
            CG_REQUIRE(Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(toRead, 1))));
            rhi::BufferCopy back{};
            back.source = outBuffer;
            back.destination = readback;
            back.sizeBytes = kResultBytes;
            CG_REQUIRE(Step(__LINE__, list.CopyBuffer(back)));
            CG_REQUIRE(Step(__LINE__, list.End()));
            const rhi::CommandListHandle handle = list.Handle();
            rhi::SubmitDesc submit{};
            submit.queue = rhi::QueueKind::Graphics;
            submit.commandLists = Span<const rhi::CommandListHandle>(&handle, 1);
            CG_REQUIRE(Step(__LINE__, device.Submit(submit)));
            CG_REQUIRE(Step(__LINE__, device.WaitIdle()));
            u32 results[kTextures * kSamples] = {};
            CG_REQUIRE(Step(__LINE__, device.ReadBuffer(readback, 0,
                                                        Span<Byte>(reinterpret_cast<Byte*>(results),
                                                                   sizeof(results)))));
            for (u32 t = 0; t < kTextures; ++t)
            {
                const u32 dim = paramWords[t * 2u + 1u];
                for (u32 s = 0; s < kSamples; ++s)
                {
                    const u32 x = (s * 37u + t * 11u) % dim;
                    const u32 y = (s * 53u + t * 5u) % dim;
                    u32 expected = 0;
                    for (u32 c = 0; c < 4; ++c)
                    {
                        const usize offset = (static_cast<usize>(y) * dim + x) * 4u + c;
                        expected |= static_cast<u32>(PatternByte(t, absoluteMip[t], offset))
                                    << (8u * c);
                    }
                    ++samplesChecked;
                    const bool wrong = results[t * kSamples + s] != expected;
                    mismatches += wrong ? 1u : 0u;
                    if (wrong && mismatches <= 6)
                    {
                        std::printf("  frame %u texture %u mip %u (resident %u) sample %u: got "
                                    "%08x want %08x\n",
                                    f, t, absoluteMip[t], streamer.ResidentLevel(handles[t]), s,
                                    results[t * kSamples + s], expected);
                    }
                }
            }
            (void)device.DestroyBindGroup(group.Value());
            for (const rhi::TextureViewHandle view : views)
            {
                (void)device.DestroyTextureView(view);
            }
        }

        std::sort(streamMs.begin(), streamMs.end());
        const double p99 = streamMs[streamMs.size() * 99u / 100u];
        const StreamingStats& stats = streamer.Stats();
        std::printf("adapter %u: %llu samples, %llu wrong; budget %u of %u tiles; sharpest level "
                    "%u; streaming frame p99 %.2f ms, worst %.2f ms; %u uploads, %u evictions\n",
                    adapterIndex, static_cast<unsigned long long>(samplesChecked),
                    static_cast<unsigned long long>(mismatches), budgetTiles, totalTiles, sharpest,
                    p99, worstStreamMs, pool.Stats().uploads, pool.Stats().evictions);
        CG_EXPECT_EQ(mismatches, 0ull);
        CG_EXPECT_TRUE(sharpest == 0u);
        CG_EXPECT_TRUE(pool.Stats().evictions > 0u);
        CG_EXPECT_TRUE(stats.usedPages <= budgetTiles);
        // The streaming work (update, copies, submit) must be steady: the 99th
        // percentile frame stays inside a generous bound. The single worst frame
        // is printed but not asserted, because one scheduler hiccup on a busy
        // machine (seen once at 182 ms under parallel tests) says nothing about
        // the streamer.
#if defined(CG_DEBUG) && CG_DEBUG
        constexpr double kP99BoundMs = 250.0;  // unoptimized code and CRT checks
#else
        constexpr double kP99BoundMs = 50.0;
#endif
        CG_EXPECT_TRUE(p99 < kP99BoundMs);

        (void)device.DestroyComputePipeline(pipeline.Value());
        (void)device.DestroyPipelineLayout(pipelineLayout.Value());
        (void)device.DestroyBindGroupLayout(groupLayout.Value());
        (void)device.DestroyShader(shader.Value());
        for (const rhi::BufferHandle buffer : {paramStage, paramBuffer, outBuffer, readback})
        {
            (void)device.DestroyBuffer(buffer);
        }
        for (const TextureHandle handle : handles)
        {
            pool.RemoveTexture(streamer, handle);
        }
        (void)streamer.Update(pool);
        streamer.Shutdown();
        pool.Shutdown();
    }
    if (tested == 0)
    {
        CG_SKIP("no Direct3D 12 adapter with reserved textures");
    }
}
