// BC6H against the real decoders (Phase 33): the CPU decoder in `cyber_assets`
// claims to be bit-exact with the Direct3D 11 specification, and the hardware
// block decoder of every local Direct3D 12 adapter (the Intel HD Graphics 530
// and WARP here) is the reference that says whether it is. Random blocks of
// every one of the 14 modes, and blocks the encoder produced from an HDR image,
// are uploaded as BC6H textures, loaded by a compute shader, and compared with
// `DecodeBc6hBlock` bit for bit, for both the unsigned and signed formats.

#include <cyber/assets/bc_codec.h>
#include <cyber/core/log.h>
#include <cyber/core/log_sinks.h>
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

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace
{

using cyber::core::Byte;
using cyber::core::f32;
using cyber::core::u16;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::usize;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
namespace rhi = cyber::rhi;
namespace assets = cyber::assets;

constexpr u32 kBlocksX = 64;
constexpr u32 kModeCount = 14;
constexpr u32 kBlocksPerMode = 64;
constexpr u32 kEncodedBlocks = 256;
constexpr u32 kBlocksY = (kModeCount * kBlocksPerMode + kEncodedBlocks) / kBlocksX;
constexpr u32 kWidth = kBlocksX * 4;
constexpr u32 kHeight = kBlocksY * 4;

// The mode value and its width in bits, in table order (mode 1 .. 14).
struct ModeBits
{
    u32 value;
    u32 bits;
};
constexpr ModeBits kModeBits[kModeCount] = {{0x00, 2}, {0x01, 2}, {0x02, 5}, {0x06, 5}, {0x0A, 5},
                                            {0x0E, 5}, {0x12, 5}, {0x16, 5}, {0x1A, 5}, {0x1E, 5},
                                            {0x03, 5}, {0x07, 5}, {0x0B, 5}, {0x0F, 5}};

u32 NextRandom(u32& state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}

// Random blocks of each mode, then blocks the encoder made from a smooth HDR
// image: the first `kModeCount * kBlocksPerMode` blocks are random.
std::vector<Byte> MakeBlocks(bool isSigned)
{
    std::vector<Byte> blocks(static_cast<usize>(kBlocksX) * kBlocksY * 16u);
    u32 state = isSigned ? 0x51ED270Bu : 0x1234ABCDu;
    for (u32 mode = 0; mode < kModeCount; ++mode)
    {
        for (u32 n = 0; n < kBlocksPerMode; ++n)
        {
            Byte* block = blocks.data() + (static_cast<usize>(mode) * kBlocksPerMode + n) * 16u;
            for (u32 i = 0; i < 16; ++i)
            {
                block[i] = static_cast<Byte>(NextRandom(state) >> 24u);
            }
            const u32 bits = kModeBits[mode].bits;
            block[0] = static_cast<Byte>((static_cast<u32>(block[0]) & ~((1u << bits) - 1u)) |
                                         kModeBits[mode].value);
        }
    }
    // Encoder output from a wide-range image.
    assets::BcEncodeSettings settings;
    settings.quality = assets::BcQuality::High;
    for (u32 n = 0; n < kEncodedBlocks; ++n)
    {
        u16 texels[48];
        const f32 scale = std::exp(static_cast<f32>(n % 17u) * 0.9f - 6.0f);
        for (u32 t = 0; t < 16; ++t)
        {
            for (u32 c = 0; c < 3; ++c)
            {
                f32 value = scale * (1.0f + static_cast<f32>(NextRandom(state) >> 20u) / 4096.0f *
                                                static_cast<f32>((n / 17u) % 5u));
                if (isSigned && ((n + c) & 1u) != 0)
                {
                    value = -value;
                }
                texels[t * 3u + c] = assets::FloatToHalf(value);
            }
        }
        assets::EncodeBc6hBlock(
            texels, blocks.data() + (static_cast<usize>(kModeCount) * kBlocksPerMode + n) * 16u,
            settings, isSigned);
    }
    return blocks;
}

const char* const kShader = R"hlsl(
Texture2D<float4> Source : register(t0, space0);
RWStructuredBuffer<float4> Output : register(u0, space0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint x = id.x % 256u;
    uint y = id.x / 256u;
    Output[id.x] = Source.Load(int3(x, y, 0));
}
)hlsl";

template <class S>
bool Step(int line, const S& status)
{
    if (!status)
    {
        std::printf("step at line %d failed: %s\n", line, status.Error().message);
    }
    return status.IsOk();
}

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

// Loads every texel of a BC6H texture holding `blocks` on `device` and returns
// the float4s. False when any step of the GPU work fails.
bool DecodeOnGpu(rhi::Device& device, rhi::Format format, const std::vector<Byte>& blocks,
                 std::vector<f32>& texels)
{
    const u64 rowBytes = static_cast<u64>(kBlocksX) * 16u;  // a multiple of 256
    const u64 resultBytes = static_cast<u64>(kWidth) * kHeight * 16u;

    rhi::BufferDesc stageDesc{};
    stageDesc.usage = rhi::BufferUsage::HostVisible | rhi::BufferUsage::CopySource;
    stageDesc.sizeBytes = blocks.size();
    stageDesc.label = "bc6h_stage";
    auto stage = device.CreateBuffer(stageDesc);
    rhi::BufferDesc outDesc{};
    outDesc.usage = rhi::BufferUsage::Storage | rhi::BufferUsage::CopySource;
    outDesc.sizeBytes = resultBytes;
    outDesc.elementStride = 16;
    outDesc.label = "bc6h_out";
    auto out = device.CreateBuffer(outDesc);
    rhi::BufferDesc readDesc{};
    readDesc.usage = rhi::BufferUsage::Readback | rhi::BufferUsage::CopyDestination;
    readDesc.sizeBytes = resultBytes;
    readDesc.label = "bc6h_readback";
    auto readback = device.CreateBuffer(readDesc);
    rhi::TextureDesc textureDesc{};
    textureDesc.format = format;
    textureDesc.usage = rhi::TextureUsage::Sampled | rhi::TextureUsage::CopyDestination;
    textureDesc.width = kWidth;
    textureDesc.height = kHeight;
    textureDesc.label = "bc6h_texture";
    auto texture = device.CreateTexture(textureDesc);
    if (!stage || !out || !readback || !texture)
    {
        std::printf("resource creation failed\n");
        return false;
    }
    rhi::TextureViewDesc viewDesc{};
    viewDesc.texture = texture.Value();
    viewDesc.format = format;
    viewDesc.range.mipLevelCount = 1;
    auto view = device.CreateTextureView(viewDesc);

    std::string source = kShader;
    rhi::ShaderDesc shaderDesc{};
    shaderDesc.stage = rhi::ShaderStage::Compute;
    shaderDesc.format = rhi::ShaderBytecodeFormat::HlslSource;
    shaderDesc.code = Span<const Byte>(reinterpret_cast<const Byte*>(source.data()), source.size());
    shaderDesc.entryPoint = "main";
    shaderDesc.label = "bc6h_load";
    auto shader = device.CreateShader(shaderDesc);
    const rhi::ShaderStageMask compute = rhi::ShaderStageBit(rhi::ShaderStage::Compute);
    rhi::BindGroupLayoutEntry entries[2]{};
    entries[0] = {0, rhi::BindingType::SampledTexture, compute, 1, 0, false};
    entries[1] = {1, rhi::BindingType::StorageBuffer, compute, 1, 0, false};
    rhi::BindGroupLayoutDesc layoutDesc{};
    layoutDesc.entries = Span<const rhi::BindGroupLayoutEntry>(entries, 2);
    auto groupLayout = device.CreateBindGroupLayout(layoutDesc);
    if (!view || !shader || !groupLayout)
    {
        std::printf("view/shader/layout creation failed\n");
        return false;
    }
    rhi::PipelineLayoutDesc pipelineLayoutDesc{};
    pipelineLayoutDesc.groups[static_cast<u32>(rhi::BindGroupSlot::PerFrame)] = groupLayout.Value();
    auto pipelineLayout = device.CreatePipelineLayout(pipelineLayoutDesc);
    if (!pipelineLayout)
    {
        return false;
    }
    rhi::ComputePipelineDesc pipelineDesc{};
    pipelineDesc.layout = pipelineLayout.Value();
    pipelineDesc.computeShader = shader.Value();
    auto pipeline = device.CreateComputePipeline(pipelineDesc);
    if (!pipeline)
    {
        std::printf("pipeline: %s\n", pipeline.Error().message);
        return false;
    }
    rhi::BindingResource resources[2]{};
    resources[0].type = rhi::BindingType::SampledTexture;
    resources[0].textureView = view.Value();
    resources[1].type = rhi::BindingType::StorageBuffer;
    resources[1].buffer = out.Value();
    rhi::BindGroupDesc groupDesc{};
    groupDesc.layout = groupLayout.Value();
    groupDesc.resources = Span<const rhi::BindingResource>(resources, 2);
    auto group = device.CreateBindGroup(groupDesc);
    if (!group)
    {
        return false;
    }

    bool ok = Step(__LINE__, device.WriteBuffer(stage.Value(), 0,
                                                Span<const Byte>(blocks.data(), blocks.size())));
    rhi::CommandListDesc listDesc{};
    listDesc.queue = rhi::QueueKind::Graphics;
    auto createdList = device.CreateCommandList(listDesc);
    ok = ok && createdList.IsOk();
    if (ok)
    {
        rhi::CommandList list = std::move(createdList.Value());
        ok = Step(__LINE__, list.Begin());
        rhi::SubresourceRange range{};
        range.mipLevelCount = 1;
        range.layerCount = 1;
        const rhi::BufferBarrier stageBarrier{stage.Value(), device.BufferStateOf(stage.Value()),
                                              rhi::ResourceState::CopySource};
        const rhi::TextureBarrier toCopy{texture.Value(), range,
                                         device.TextureStateOf(texture.Value(), 0, 0),
                                         rhi::ResourceState::CopyDestination};
        ok = ok && Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(&stageBarrier, 1),
                                               Span<const rhi::TextureBarrier>(&toCopy, 1)));
        rhi::BufferTextureCopy copy{};
        copy.buffer = stage.Value();
        copy.bufferRowPitch = static_cast<u32>(rowBytes);
        copy.texture = texture.Value();
        copy.width = kWidth;
        copy.height = kHeight;
        ok = ok && Step(__LINE__, list.CopyBufferToTexture(copy));
        const rhi::TextureBarrier toSample{texture.Value(), range,
                                           rhi::ResourceState::CopyDestination,
                                           rhi::ResourceState::ShaderRead};
        const rhi::BufferBarrier outBarrier{out.Value(), device.BufferStateOf(out.Value()),
                                            rhi::ResourceState::ShaderWrite};
        ok = ok && Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(&outBarrier, 1),
                                               Span<const rhi::TextureBarrier>(&toSample, 1)));
        ok = ok && Step(__LINE__, list.SetPipeline(pipeline.Value())) &&
             Step(__LINE__, list.SetBindGroup(rhi::BindGroupSlot::PerFrame, group.Value()));
        rhi::DispatchArguments dispatch{};
        dispatch.groupCountX = kWidth * kHeight / 64u;
        ok = ok && Step(__LINE__, list.Dispatch(dispatch));
        const rhi::BufferBarrier toRead[] = {
            {out.Value(), rhi::ResourceState::ShaderWrite, rhi::ResourceState::CopySource},
            {readback.Value(), device.BufferStateOf(readback.Value()),
             rhi::ResourceState::CopyDestination}};
        ok = ok && Step(__LINE__, list.Barrier(Span<const rhi::BufferBarrier>(toRead, 2)));
        rhi::BufferCopy back{};
        back.source = out.Value();
        back.destination = readback.Value();
        back.sizeBytes = resultBytes;
        ok = ok && Step(__LINE__, list.CopyBuffer(back)) && Step(__LINE__, list.End());
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
            texels.assign(static_cast<usize>(kWidth) * kHeight * 4u, 0.0f);
            ok = Step(__LINE__, device.ReadBuffer(readback.Value(), 0,
                                                  Span<Byte>(reinterpret_cast<Byte*>(texels.data()),
                                                             static_cast<usize>(resultBytes))));
        }
        (void)device.WaitIdle();
    }
    (void)device.DestroyBindGroup(group.Value());
    (void)device.DestroyComputePipeline(pipeline.Value());
    (void)device.DestroyPipelineLayout(pipelineLayout.Value());
    (void)device.DestroyBindGroupLayout(groupLayout.Value());
    (void)device.DestroyShader(shader.Value());
    (void)device.DestroyTextureView(view.Value());
    (void)device.DestroyTexture(texture.Value());
    (void)device.DestroyBuffer(stage.Value());
    (void)device.DestroyBuffer(out.Value());
    (void)device.DestroyBuffer(readback.Value());
    return ok;
}

}  // namespace

CG_TEST(bc6h_cpu_decoder_matches_the_hardware_decoder_on_every_mode)
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

        for (const bool isSigned : {false, true})
        {
            const rhi::Format format = isSigned ? rhi::Format::BC6HSfloat : rhi::Format::BC6HUfloat;
            const std::vector<Byte> blocks = MakeBlocks(isSigned);
            std::vector<f32> gpu;
            CG_REQUIRE(DecodeOnGpu(device, format, blocks, gpu));

            u32 wrongByMode[kModeCount + 1] = {};
            u32 firstWrong[kModeCount + 1];
            for (u32& value : firstWrong)
            {
                value = 0xFFFFFFFFu;
            }
            for (u32 blockIndex = 0; blockIndex < kBlocksX * kBlocksY; ++blockIndex)
            {
                const u32 mode = blockIndex < kModeCount * kBlocksPerMode
                                     ? blockIndex / kBlocksPerMode
                                     : kModeCount;
                u16 decoded[48];
                assets::DecodeBc6hBlock(blocks.data() + static_cast<usize>(blockIndex) * 16u,
                                        decoded, isSigned);
                const u32 bx = blockIndex % kBlocksX;
                const u32 by = blockIndex / kBlocksX;
                bool wrong = false;
                for (u32 t = 0; t < 16 && !wrong; ++t)
                {
                    const u32 x = bx * 4u + (t % 4u);
                    const u32 y = by * 4u + (t / 4u);
                    for (u32 c = 0; c < 3; ++c)
                    {
                        const f32 want = assets::HalfToFloat(decoded[t * 3u + c]);
                        const f32 got = gpu[(static_cast<usize>(y) * kWidth + x) * 4u + c];
                        if (std::memcmp(&want, &got, sizeof(f32)) != 0 && !(want == got))
                        {
                            wrong = true;
                            break;
                        }
                    }
                }
                if (wrong)
                {
                    ++wrongByMode[mode];
                    if (firstWrong[mode] == 0xFFFFFFFFu)
                    {
                        firstWrong[mode] = blockIndex;
                    }
                }
            }
            u32 total = 0;
            for (u32 mode = 0; mode <= kModeCount; ++mode)
            {
                total += wrongByMode[mode];
                if (wrongByMode[mode] != 0)
                {
                    std::printf("  adapter %u %s: %s %u differ (first block %u)\n", adapterIndex,
                                isSigned ? "signed" : "unsigned",
                                mode == kModeCount ? "encoded blocks" : "mode", wrongByMode[mode],
                                firstWrong[mode]);
                    if (mode < kModeCount)
                    {
                        std::printf("    mode %u\n", mode + 1);
                    }
                }
            }
            std::printf("adapter %u %s: %u of %u blocks differ from the hardware decoder\n",
                        adapterIndex, isSigned ? "signed" : "unsigned", total, kBlocksX * kBlocksY);
            CG_EXPECT_EQ(total, 0u);
        }
        CG_EXPECT_EQ(device.ReportLeaks(), 0u);
    }
    if (tested == 0)
    {
        CG_SKIP("no Direct3D 12 adapter could be created");
    }
}
