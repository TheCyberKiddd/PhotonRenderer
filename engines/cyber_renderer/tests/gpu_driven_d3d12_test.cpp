#include <cyber/core/log.h>
#include <cyber/core/log_sinks.h>
#include <cyber/jobs/job_system.h>
#include <cyber/math/aabb.h>
#include <cyber/math/constants.h>
#include <cyber/math/mat4.h>
#include <cyber/math/vec3.h>
#include <cyber/math/vec4.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/renderer/clustered_lighting.h>
#include <cyber/renderer/gpu_culling_pipeline.h>
#include <cyber/renderer/gpu_driven.h>
#include <cyber/renderer/gpu_light_culling.h>
#include <cyber/rhi/backend.h>
#include <cyber/rhi/binding.h>
#include <cyber/rhi/buffer.h>
#include <cyber/rhi/capabilities.h>
#include <cyber/rhi/command_list.h>
#include <cyber/rhi/d3d12.h>
#include <cyber/rhi/device.h>
#include <cyber/rhi/pipeline.h>
#include <cyber/rhi/render_pass.h>
#include <cyber/rhi/resource_state.h>
#include <cyber/rhi/shader.h>
#include <cyber/rhi/submission.h>
#include <cyber/rhi/texture.h>
#include <cyber/scene/scene.h>
#include <cyber/test/test.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>

using namespace cyber::renderer;
using namespace cyber::rhi;
using cyber::core::f32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;
using cyber::core::usize;
using cyber::math::Aabb;
using cyber::math::Mat4;
using cyber::math::Vec3;
using cyber::math::Vec4;
using cyber::memory::Byte;
using cyber::memory::DynamicArray;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;

namespace
{

constexpr f32 kFov = cyber::math::kPi / 3.0f;
constexpr u32 kViewportWidth = 1280;
constexpr u32 kViewportHeight = 720;
constexpr f32 kNear = 0.1f;
constexpr f32 kFar = 1500.0f;
// The roadmap's acceptance scene.
constexpr u32 kMillion = 1000000;
// The point target: one texel per instance identifier.
constexpr u32 kTargetSize = 1024;
constexpr u32 kTargetRowPitch = kTargetSize * 4;

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

struct Adapter
{
    u32 index = 0;
    bool software = false;
    char name[kAdapterNameBytes]{};
};

// Every Direct3D 12 adapter the machine exposes, hardware first.
[[nodiscard]] u32 ListAdapters(Adapter (&out)[8]) noexcept
{
    const Backend* backend = D3D12Backend();
    if (backend == nullptr)
    {
        return 0;
    }
    AdapterInfo infos[8]{};
    const u32 count = std::min<u32>(backend->enumerateAdapters(infos, 8), 8);
    u32 written = 0;
    for (u32 pass = 0; pass < 2; ++pass)
    {
        for (u32 index = 0; index < count; ++index)
        {
            const bool software = infos[index].vendorId == 0x1414;
            if (software != (pass == 1))
            {
                continue;
            }
            out[written].index = index;
            out[written].software = software;
            std::snprintf(out[written].name, sizeof(out[written].name), "%s", infos[index].name);
            ++written;
        }
    }
    return written;
}

[[nodiscard]] bool OpenDevice(const Adapter& adapter, Device& out, HeapAllocator& heap) noexcept
{
    DeviceDesc desc{};
    desc.backend = D3D12Backend();
    desc.adapterIndex = adapter.index;
    auto created = Device::Create(desc, heap);
    if (!created)
    {
        return false;
    }
    out = std::move(created.Value());
    return true;
}

// The build guide counts any debug-layer message as a failure. The backend
// drains the layer's queue on the `Rhi` channel when the device goes away;
// this watches that channel for the lifetime of the scope.
struct DebugLayerWatch
{
    u32 records = 0;
    char last[cyber::core::kMaxLogMessageLength]{};
};

void WatchLog(void* userData, const cyber::core::LogRecord& record) noexcept
{
    auto* watch = static_cast<DebugLayerWatch*>(userData);
    if (record.channel != cyber::core::LogChannel::Rhi ||
        record.level < cyber::core::LogLevel::Warning)
    {
        return;
    }
    watch->records += 1;
    std::snprintf(watch->last, sizeof(watch->last), "%.220s", record.message);
}

// ---------------------------------------------------------------------------
// Scenes
// ---------------------------------------------------------------------------

struct Random
{
    u64 state = 0;

    explicit Random(u64 seed) noexcept : state(seed * 0x9E3779B97F4A7C15ULL + 1) {}

    [[nodiscard]] u32 Next() noexcept
    {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<u32>(state >> 33U);
    }

    [[nodiscard]] f32 Range(f32 low, f32 high) noexcept
    {
        return low +
               (high - low) * static_cast<f32>(Next() & 0xFFFFFFU) / static_cast<f32>(0x1000000);
    }
};

[[nodiscard]] Mat4 ViewProjection(Vec3 eye, Vec3 target) noexcept
{
    const Mat4 view = Mat4::LookAt(eye, target, Vec3{0.0f, 1.0f, 0.0f});
    const Mat4 projection = Mat4::PerspectiveReverseZ(
        kFov, static_cast<f32>(kViewportWidth) / static_cast<f32>(kViewportHeight), kNear, kFar);
    return projection * view;
}

constexpr u32 kGroups = 16;

// Sixteen draw groups with up to four LODs. Every LOD draws index 0 once:
// the point pipeline below places a point by instance identifier, so the
// geometry only has to exist.
[[nodiscard]] bool AddPointGroups(GpuDrivenScene& scene) noexcept
{
    for (u32 group = 0; group < kGroups; ++group)
    {
        const u32 lodCount = 1 + group % kGpuDrivenMaxLods;
        GpuDrawLod lods[kGpuDrivenMaxLods]{};
        for (u32 lod = 0; lod < lodCount; ++lod)
        {
            lods[lod].indexCount = 1;
            lods[lod].firstIndex = lod;
            lods[lod].vertexOffset = 0;
        }
        GpuDrawGroupDesc desc{};
        desc.lods = Span<const GpuDrawLod>(lods, lodCount);
        desc.lodPixelThresholds[0] = 120.0f;
        desc.lodPixelThresholds[1] = 30.0f;
        desc.lodPixelThresholds[2] = 6.0f;
        if (!scene.AddDrawGroup(desc))
        {
            return false;
        }
    }
    return true;
}

// `count` boxes spread through a large volume around the camera, recorded as
// translation and extent so Phase 22's `TransformBounds` reproduces the same
// world bounds exactly.
struct World
{
    explicit World(cyber::memory::Allocator& allocator) noexcept
        : scene(allocator), translations(allocator), extents(allocator)
    {
    }

    GpuDrivenScene scene;
    DynamicArray<Vec3> translations;
    DynamicArray<Vec3> extents;
};

[[nodiscard]] bool BuildWorld(World& world, u32 count, u64 seed) noexcept
{
    Random random(seed);
    if (!AddPointGroups(world.scene) || !world.scene.Reserve(count) ||
        !world.translations.TryReserve(count) || !world.extents.TryReserve(count))
    {
        return false;
    }
    for (u32 index = 0; index < count; ++index)
    {
        const Vec3 translation{random.Range(-800.0f, 800.0f), random.Range(-120.0f, 120.0f),
                               random.Range(-1200.0f, 300.0f)};
        const f32 size = random.Range(0.05f, 4.0f);
        const Vec3 extent{size, size * random.Range(0.3f, 1.5f), size * random.Range(0.3f, 1.5f)};
        (void)world.translations.TryPushBack(translation);
        (void)world.extents.TryPushBack(extent);
        if (!world.scene.AddInstance(index % kGroups, Aabb::FromCenterExtents(translation, extent),
                                     index))
        {
            return false;
        }
    }
    return static_cast<bool>(world.scene.Finalize());
}

[[nodiscard]] GpuCullView DefaultView() noexcept
{
    const Vec3 eye{0.0f, 4.0f, 120.0f};
    return MakeGpuCullView(ViewProjection(eye, Vec3{30.0f, 0.0f, -400.0f}), eye, kFov,
                           kViewportHeight, kNear);
}

// Instances a scene gets on an adapter. WARP runs every test through the same
// code path, but under the debug layer's GPU-based validation it needs about
// two minutes per million-instance frame, so it gets a tenth of the scene.
// The million-instance acceptance runs on hardware adapters.
[[nodiscard]] u32 InstancesFor(const Adapter& adapter, u32 hardwareCount) noexcept
{
    return adapter.software ? std::max<u32>(hardwareCount / 10, 1000) : hardwareCount;
}

// ---------------------------------------------------------------------------
// A point pipeline that writes one texel per drawn instance
// ---------------------------------------------------------------------------

// Position is unused but present: binding 0 is the per-vertex stream every
// indexed draw reads. Binding 1 is the culler's visible-instance stream.
constexpr char kPointVertexSource[] = R"HLSL(
struct Input
{
    float3 position : TEXCOORD0;
    uint instanceId : TEXCOORD1;
};

float4 main(Input input) : SV_Position
{
    uint x = input.instanceId % 1024;
    uint y = input.instanceId / 1024;
    float2 ndc = float2((x + 0.5) / 512.0 - 1.0, 1.0 - (y + 0.5) / 512.0);
    return float4(ndc + input.position.xy * 0.0, 0.5, 1.0);
}
)HLSL";

constexpr char kPointPixelSource[] = R"HLSL(
float4 main(float4 position : SV_Position) : SV_Target
{
    return float4(1.0, 1.0, 1.0, 1.0);
}
)HLSL";

[[nodiscard]] Span<const Byte> Text(const char* text) noexcept
{
    return Span<const Byte>(reinterpret_cast<const Byte*>(text), std::strlen(text));
}

struct PointTarget
{
    Device* device = nullptr;
    PipelineLayoutHandle layout{};
    ShaderHandle vertex{};
    ShaderHandle pixel{};
    PipelineHandle pipeline{};
    BufferHandle vertices{};
    BufferHandle indices{};
    BufferHandle staging{};
    TextureHandle color{};
    TextureViewHandle colorView{};
    BufferHandle readback{};
    bool uploaded = false;
    ResourceState colorState = ResourceState::Undefined;

    PointTarget() noexcept = default;
    PointTarget(const PointTarget&) = delete;
    PointTarget& operator=(const PointTarget&) = delete;

    ~PointTarget()
    {
        if (device == nullptr)
        {
            return;
        }
        (void)device->DestroyGraphicsPipeline(pipeline);
        (void)device->DestroyPipelineLayout(layout);
        (void)device->DestroyShader(vertex);
        (void)device->DestroyShader(pixel);
        (void)device->DestroyBuffer(vertices);
        (void)device->DestroyBuffer(indices);
        (void)device->DestroyBuffer(staging);
        (void)device->DestroyTextureView(colorView);
        (void)device->DestroyTexture(color);
        (void)device->DestroyBuffer(readback);
    }
};

[[nodiscard]] bool BuildPointTarget(Device& device, PointTarget& target) noexcept
{
    target.device = &device;
    PipelineLayoutDesc layoutDesc{};
    auto layout = device.CreatePipelineLayout(layoutDesc);
    ShaderDesc vertexDesc{};
    vertexDesc.stage = ShaderStage::Vertex;
    vertexDesc.code = Text(kPointVertexSource);
    vertexDesc.entryPoint = "main";
    auto vertex = device.CreateShader(vertexDesc);
    ShaderDesc pixelDesc{};
    pixelDesc.stage = ShaderStage::Pixel;
    pixelDesc.code = Text(kPointPixelSource);
    pixelDesc.entryPoint = "main";
    auto pixel = device.CreateShader(pixelDesc);
    if (!layout || !vertex || !pixel)
    {
        return false;
    }
    target.layout = layout.Value();
    target.vertex = vertex.Value();
    target.pixel = pixel.Value();

    VertexBinding bindings[2]{};
    bindings[0].strideBytes = 12;
    bindings[0].rate = VertexInputRate::PerVertex;
    bindings[1].strideBytes = 4;
    bindings[1].rate = VertexInputRate::PerInstance;
    VertexAttribute attributes[2]{};
    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = Format::R32G32B32Float;
    attributes[1].location = 1;
    attributes[1].binding = kGpuDrivenInstanceStreamBinding;
    attributes[1].format = Format::R32Uint;
    ColorAttachmentState attachment{};
    attachment.format = Format::RGBA8Unorm;
    GraphicsPipelineDesc pipelineDesc{};
    pipelineDesc.layout = target.layout;
    pipelineDesc.vertexShader = target.vertex;
    pipelineDesc.pixelShader = target.pixel;
    pipelineDesc.vertexLayout.bindings = Span<const VertexBinding>(bindings, 2);
    pipelineDesc.vertexLayout.attributes = Span<const VertexAttribute>(attributes, 2);
    pipelineDesc.topology = PrimitiveTopology::PointList;
    pipelineDesc.rasterizer.cullMode = CullMode::None;
    // Points do not overlap, so there is nothing for a depth test to decide.
    pipelineDesc.depthStencil.depthTestEnabled = false;
    pipelineDesc.depthStencil.depthWriteEnabled = false;
    pipelineDesc.colorAttachments = Span<const ColorAttachmentState>(&attachment, 1);
    auto pipeline = device.CreateGraphicsPipeline(pipelineDesc);
    if (!pipeline)
    {
        std::printf("point pipeline: %s\n", pipeline.Error().message);
        return false;
    }
    target.pipeline = pipeline.Value();

    BufferDesc vertexBuffer{};
    vertexBuffer.usage = BufferUsage::Vertex | BufferUsage::CopyDestination;
    vertexBuffer.sizeBytes = 256;
    BufferDesc indexBuffer{};
    indexBuffer.usage = BufferUsage::Index | BufferUsage::CopyDestination;
    indexBuffer.sizeBytes = 256;
    BufferDesc stagingBuffer{};
    stagingBuffer.usage = BufferUsage::HostVisible | BufferUsage::CopySource;
    stagingBuffer.sizeBytes = 512;
    BufferDesc readbackBuffer{};
    readbackBuffer.usage = BufferUsage::Readback | BufferUsage::CopyDestination;
    readbackBuffer.sizeBytes = static_cast<u64>(kTargetRowPitch) * kTargetSize;
    auto vertices = device.CreateBuffer(vertexBuffer);
    auto indices = device.CreateBuffer(indexBuffer);
    auto staging = device.CreateBuffer(stagingBuffer);
    auto readback = device.CreateBuffer(readbackBuffer);
    TextureDesc colorDesc{};
    colorDesc.format = Format::RGBA8Unorm;
    colorDesc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
    colorDesc.width = kTargetSize;
    colorDesc.height = kTargetSize;
    auto color = device.CreateTexture(colorDesc);
    if (!vertices || !indices || !staging || !readback || !color)
    {
        return false;
    }
    target.vertices = vertices.Value();
    target.indices = indices.Value();
    target.staging = staging.Value();
    target.readback = readback.Value();
    target.color = color.Value();
    TextureViewDesc viewDesc{};
    viewDesc.texture = target.color;
    viewDesc.format = Format::RGBA8Unorm;
    auto view = device.CreateTextureView(viewDesc);
    if (!view)
    {
        return false;
    }
    target.colorView = view.Value();
    const f32 zeros[3] = {0.0f, 0.0f, 0.0f};
    const u32 indexData[4] = {0, 0, 0, 0};
    return device
               .WriteBuffer(target.staging, 0,
                            Span<const Byte>(reinterpret_cast<const Byte*>(zeros), sizeof(zeros)))
               .IsOk() &&
           device
               .WriteBuffer(
                   target.staging, 256,
                   Span<const Byte>(reinterpret_cast<const Byte*>(indexData), sizeof(indexData)))
               .IsOk();
}

// Uploads the point geometry on first use, opens a pass that clears the
// target, and binds the point pipeline and its streams.
[[nodiscard]] bool BeginPointPass(PointTarget& target, CommandList& list) noexcept
{
    if (!target.uploaded)
    {
        const BufferBarrier upload[] = {
            {target.staging, ResourceState::Undefined, ResourceState::CopySource},
            {target.vertices, ResourceState::Undefined, ResourceState::CopyDestination},
            {target.indices, ResourceState::Undefined, ResourceState::CopyDestination},
            {target.readback, ResourceState::Undefined, ResourceState::CopyDestination},
        };
        BufferCopy vertexCopy{target.staging, 0, target.vertices, 0, 12};
        BufferCopy indexCopy{target.staging, 256, target.indices, 0, 16};
        const BufferBarrier ready[] = {
            {target.vertices, ResourceState::CopyDestination, ResourceState::VertexBuffer},
            {target.indices, ResourceState::CopyDestination, ResourceState::IndexBuffer},
        };
        if (!list.Barrier(Span<const BufferBarrier>(upload, 4)) || !list.CopyBuffer(vertexCopy) ||
            !list.CopyBuffer(indexCopy) || !list.Barrier(Span<const BufferBarrier>(ready, 2)))
        {
            return false;
        }
        target.uploaded = true;
    }
    const TextureBarrier toTarget{target.color, SubresourceRange{}, target.colorState,
                                  ResourceState::RenderTarget};
    if (!list.Barrier(Span<const BufferBarrier>{}, Span<const TextureBarrier>(&toTarget, 1)))
    {
        return false;
    }
    target.colorState = ResourceState::RenderTarget;
    ColorAttachment attachment{};
    attachment.view = target.colorView;
    attachment.loadOp = LoadOp::Clear;
    attachment.storeOp = StoreOp::Store;
    attachment.clearValue = ClearColorValue::Of(0.0f, 0.0f, 0.0f, 0.0f);
    RenderPassDesc pass{};
    pass.colorAttachments = Span<const ColorAttachment>(&attachment, 1);
    pass.label = "GPU-driven point pass";
    return list.BeginRenderPass(pass).IsOk() && list.SetPipeline(target.pipeline).IsOk() &&
           list.SetViewport(
                   Viewport::Of(static_cast<f32>(kTargetSize), static_cast<f32>(kTargetSize)))
               .IsOk() &&
           list.SetScissor(ScissorRect{0, 0, kTargetSize, kTargetSize}).IsOk() &&
           list.SetVertexBuffer(0, target.vertices).IsOk() &&
           list.SetIndexBuffer(target.indices, 0, IndexFormat::Uint32).IsOk();
}

[[nodiscard]] bool EndPointPass(PointTarget& target, CommandList& list, bool readback) noexcept
{
    if (!list.EndRenderPass())
    {
        return false;
    }
    if (!readback)
    {
        return true;
    }
    const TextureBarrier toCopy{target.color, SubresourceRange{}, ResourceState::RenderTarget,
                                ResourceState::CopySource};
    if (!list.Barrier(Span<const BufferBarrier>{}, Span<const TextureBarrier>(&toCopy, 1)))
    {
        return false;
    }
    target.colorState = ResourceState::CopySource;
    BufferTextureCopy copy{};
    copy.buffer = target.readback;
    copy.bufferRowPitch = kTargetRowPitch;
    copy.texture = target.color;
    copy.width = kTargetSize;
    copy.height = kTargetSize;
    return list.CopyTextureToBuffer(copy).IsOk();
}

// Which instance identifiers the pass lit: texel `id % 1024, id / 1024`. The
// row pitch is exactly one row, so the texel of `id` is byte `id * 4`.
[[nodiscard]] bool ReadLitIds(Device& device, const PointTarget& target,
                              cyber::memory::Allocator& allocator, DynamicArray<u8>& lit) noexcept
{
    DynamicArray<u8> pixels(allocator);
    if (!pixels.TryResize(static_cast<usize>(kTargetRowPitch) * kTargetSize) ||
        !lit.TryResize(static_cast<usize>(kTargetSize) * kTargetSize))
    {
        return false;
    }
    if (!device.ReadBuffer(target.readback, 0,
                           Span<Byte>(reinterpret_cast<Byte*>(pixels.Data()), pixels.Size())))
    {
        return false;
    }
    for (usize id = 0; id < lit.Size(); ++id)
    {
        lit[id] = pixels[id * 4] != 0 ? 1 : 0;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Submission helpers
// ---------------------------------------------------------------------------

[[nodiscard]] bool SubmitAndWait(Device& device, CommandList& list, QueueKind queue,
                                 FenceHandle fence, u64 value) noexcept
{
    const CommandListHandle handle = list.Handle();
    FenceSignal signal{fence, value};
    SubmitDesc submit{};
    submit.queue = queue;
    submit.commandLists = Span<const CommandListHandle>(&handle, 1);
    submit.signals = Span<const FenceSignal>(&signal, 1);
    return device.Submit(submit).IsOk() && device.WaitFence(fence, value).IsOk();
}

// Sorted copy of each batch's slice, so a GPU list appended with atomics and
// the reference's ascending list compare equal when they hold the same ids.
[[nodiscard]] bool SameVisibleSets(const GpuDrivenScene& scene, const GpuCullOutput& gpu,
                                   GpuCullOutput& reference) noexcept
{
    const Span<const GpuDrawBatch> batches = scene.Batches();
    if (gpu.batchCounts.Size() != batches.Size() || reference.batchCounts.Size() != batches.Size())
    {
        return false;
    }
    for (usize batch = 0; batch < batches.Size(); ++batch)
    {
        if (gpu.batchCounts[batch] != reference.batchCounts[batch])
        {
            return false;
        }
        const u32 base = batches[batch].visibleBase;
        const u32 count = gpu.batchCounts[batch];
        DynamicArray<u32> sorted(*reference.allocator);
        if (!sorted.TryResize(count))
        {
            return false;
        }
        for (u32 slot = 0; slot < count; ++slot)
        {
            sorted[slot] = gpu.visibleInstances[base + slot];
        }
        std::sort(sorted.Begin(), sorted.End());
        for (u32 slot = 0; slot < count; ++slot)
        {
            if (sorted[slot] != reference.visibleInstances[base + slot])
            {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

namespace
{

// Everything one adapter's GPU-driven run needs.
struct Harness
{
    explicit Harness(HeapAllocator& allocator) noexcept : heap(&allocator) {}

    HeapAllocator* heap = nullptr;
    Device device;
    FenceHandle fence{};
    u64 fenceValue = 0;

    [[nodiscard]] bool Open(const Adapter& adapter) noexcept
    {
        if (!OpenDevice(adapter, device, *heap))
        {
            return false;
        }
        auto created = device.CreateFence(FenceDesc{});
        if (!created)
        {
            return false;
        }
        fence = created.Value();
        return true;
    }

    [[nodiscard]] bool NewList(QueueKind queue, CommandList& out) noexcept
    {
        CommandListDesc desc{};
        desc.queue = queue;
        auto created = device.CreateCommandList(desc);
        if (!created)
        {
            return false;
        }
        out = std::move(created.Value());
        return out.Begin().IsOk();
    }

    [[nodiscard]] bool Run(CommandList& list, QueueKind queue = QueueKind::Graphics) noexcept
    {
        return list.End().IsOk() && SubmitAndWait(device, list, queue, fence, ++fenceValue);
    }
};

[[nodiscard]] GpuCullingPipelineDesc PipelineDescFor(const GpuDrivenScene& scene, u32 hiZWidth = 0,
                                                     u32 hiZHeight = 0) noexcept
{
    GpuCullingPipelineDesc desc{};
    desc.maxInstances = static_cast<u32>(scene.Instances().Size());
    desc.maxGroups = static_cast<u32>(scene.Groups().Size());
    desc.maxBatches = static_cast<u32>(scene.Batches().Size());
    desc.maxVisible = scene.VisibleCapacity();
    desc.hiZMaxWidth = hiZWidth;
    desc.hiZMaxHeight = hiZHeight;
    return desc;
}

// Instance verdicts by identifier: the LOD drawn, or `kHidden`.
constexpr u32 kHidden = 0xFFFFFFFFU;

[[nodiscard]] bool LodsById(const GpuDrivenScene& scene, const GpuCullOutput& output,
                            DynamicArray<u32>& lods) noexcept
{
    if (!lods.TryResize(scene.Instances().Size()))
    {
        return false;
    }
    for (usize index = 0; index < lods.Size(); ++index)
    {
        lods[index] = kHidden;
    }
    const Span<const GpuDrawGroup> groups = scene.Groups();
    const Span<const GpuDrawBatch> batches = scene.Batches();
    for (usize group = 0; group < groups.Size(); ++group)
    {
        for (u32 lod = 0; lod < groups[group].lodCount; ++lod)
        {
            const u32 batch = groups[group].firstBatch + lod;
            for (u32 slot = 0; slot < output.batchCounts[batch]; ++slot)
            {
                const u32 id = output.visibleInstances[batches[batch].visibleBase + slot];
                if (id >= lods.Size())
                {
                    return false;
                }
                lods[id] = lod;
            }
        }
    }
    return true;
}

// Where the GPU and the CPU disagree about an instance, the CPU's decision
// margin says whether one rounding step could explain it: the GPU's divide and
// square root are within a few ULPs, not exact. Anything else is a defect.
struct Agreement
{
    u32 mismatches = 0;
    u32 unexplained = 0;
};

[[nodiscard]] Agreement CompareDecisions(const GpuDrivenScene& scene, const GpuCullView& view,
                                         const HiZPyramid* occluders, const DynamicArray<u32>& gpu,
                                         const DynamicArray<u32>& reference) noexcept
{
    Agreement agreement{};
    const Span<const GpuCullInstance> instances = scene.Instances();
    for (usize index = 0; index < instances.Size(); ++index)
    {
        const u32 id = instances[index].instanceId;
        if (gpu[id] == reference[id])
        {
            continue;
        }
        ++agreement.mismatches;
        const GpuInstanceDecision decision = DecideGpuInstance(
            instances[index], scene.Groups()[instances[index].drawGroup], view, occluders);
        const bool explained = decision.frustumMargin < 1.0e-3f || decision.lodMargin < 1.0e-5f ||
                               (occluders != nullptr && (decision.occlusion.depthMargin < 1.0e-4f ||
                                                         decision.occlusion.pixelMargin < 1.0e-3f ||
                                                         decision.occlusion.nearMargin < 1.0e-3f));
        agreement.unexplained += explained ? 0U : 1U;
        // A failure names the first few instances, with what the CPU decided.
        if (!explained && agreement.unexplained <= 4)
        {
            std::printf("  unexplained disagreement: instance %u, GPU LOD %u, CPU LOD %u; "
                        "occlusion level %u, box depth %.9g, occluder depth %.9g\n",
                        id, gpu[id], reference[id], decision.occlusion.level,
                        decision.occlusion.boxDepth, decision.occlusion.occluderDepth);
        }
    }
    return agreement;
}

}  // namespace

// ---------------------------------------------------------------------------
// The RHI: capabilities and indirect execution
// ---------------------------------------------------------------------------

CG_TEST(gpu_driven_d3d12_reports_indirect_and_mesh_shader_capabilities)
{
    Adapter adapters[8]{};
    const u32 count = ListAdapters(adapters);
    if (count == 0)
    {
        CG_SKIP("no Direct3D 12 adapter");
    }
    HeapAllocator heap;
    for (u32 index = 0; index < count; ++index)
    {
        Device device;
        CG_REQUIRE(OpenDevice(adapters[index], device, heap));
        const Capabilities capabilities = device.Capabilities();
        CG_EXPECT_TRUE(capabilities.features.Has(Feature::IndirectDraw));
        CG_EXPECT_TRUE(capabilities.features.Has(Feature::MultiDrawIndirect));
        CG_EXPECT_TRUE(capabilities.features.Has(Feature::IndirectDrawCount));
        const GeometryPathPlan plan =
            SelectGeometryPath(capabilities, GeometryPathPreference::PreferMeshShaders);
        std::printf("adapter \"%s\": mesh shaders %s, geometry path %s (%s), compute queue %s\n",
                    adapters[index].name,
                    capabilities.features.Has(Feature::MeshShaders) ? "reported" : "not reported",
                    GeometryPathName(plan.path), GeometryFallbackReasonName(plan.reason),
                    device.Queue(QueueKind::Compute).IsValid() ? "present" : "absent");
        // Whatever the adapter reports, this build draws through the vertex
        // pipeline and says why.
        CG_EXPECT_EQ(plan.path, GeometryPath::VertexPipeline);
    }
}

// The count-buffer variant on real hardware: the GPU-side count decides how
// many draws run, the CPU-side maximum clamps it, and each draw's
// `firstInstance` offsets the per-instance stream. The plain indirect form
// runs in the same pass.
CG_TEST(gpu_driven_d3d12_indirect_count_draws_clamp_to_the_maximum)
{
    Adapter adapters[8]{};
    const u32 count = ListAdapters(adapters);
    if (count == 0)
    {
        CG_SKIP("no Direct3D 12 adapter");
    }
    HeapAllocator heap;
    DebugLayerWatch watch{};
    cyber::core::CallbackSink sink{&WatchLog, &watch};
    CG_REQUIRE(cyber::core::DefaultLogger().AddSink(&sink));
    for (u32 adapterIndex = 0; adapterIndex < count; ++adapterIndex)
    {
        Harness harness(heap);
        CG_REQUIRE(harness.Open(adapters[adapterIndex]));
        Device& device = harness.device;
        PointTarget target;
        CG_REQUIRE(BuildPointTarget(device, target));

        // Instance stream: identifiers 0..4095.
        constexpr u32 kIds = 4096;
        DynamicArray<u32> ids(heap);
        CG_REQUIRE(ids.TryResize(kIds));
        for (u32 id = 0; id < kIds; ++id)
        {
            ids[id] = id;
        }
        // Four indexed draws over disjoint identifier ranges, the count at
        // byte 128, and two plain draws at byte 160.
        const DrawIndexedArguments draws[4] = {
            {1, 10, 0, 0, 0}, {1, 20, 0, 0, 100}, {1, 30, 0, 0, 1000}, {1, 40, 0, 0, 3000}};
        const DrawArguments plainDraws[2] = {{1, 5, 0, 2000}, {1, 6, 0, 2500}};
        BufferDesc streamDesc{};
        streamDesc.usage = BufferUsage::Vertex | BufferUsage::CopyDestination;
        streamDesc.sizeBytes = kIds * sizeof(u32);
        BufferDesc argumentDesc{};
        argumentDesc.usage = BufferUsage::Indirect | BufferUsage::CopyDestination;
        argumentDesc.sizeBytes = 256;
        BufferDesc stagingDesc{};
        stagingDesc.usage = BufferUsage::HostVisible | BufferUsage::CopySource;
        stagingDesc.sizeBytes = kIds * sizeof(u32) + 256;
        auto stream = device.CreateBuffer(streamDesc);
        auto arguments = device.CreateBuffer(argumentDesc);
        auto staging = device.CreateBuffer(stagingDesc);
        CG_REQUIRE(static_cast<bool>(stream) && static_cast<bool>(arguments) &&
                   static_cast<bool>(staging));
        const u64 argumentsAt = kIds * sizeof(u32);

        // GPU count, CPU maximum, and the indexed draws that must run.
        struct Case
        {
            u32 gpuCount;
            u32 maxDraws;
            u32 expectedDraws;
        };
        const Case cases[] = {{3, 4, 3}, {9, 2, 2}, {0, 4, 0}, {4, 4, 4}};
        ResourceState streamState = ResourceState::Undefined;
        ResourceState argumentState = ResourceState::Undefined;
        for (const Case& c : cases)
        {
            u8 argumentBytes[256]{};
            std::memcpy(argumentBytes, draws, sizeof(draws));
            std::memcpy(argumentBytes + 128, &c.gpuCount, sizeof(u32));
            std::memcpy(argumentBytes + 160, plainDraws, sizeof(plainDraws));
            CG_REQUIRE(device
                           .WriteBuffer(staging.Value(), 0,
                                        Span<const Byte>(reinterpret_cast<const Byte*>(ids.Data()),
                                                         kIds * sizeof(u32)))
                           .IsOk());
            CG_REQUIRE(
                device
                    .WriteBuffer(staging.Value(), argumentsAt,
                                 Span<const Byte>(reinterpret_cast<const Byte*>(argumentBytes),
                                                  sizeof(argumentBytes)))
                    .IsOk());
            CommandList list;
            CG_REQUIRE(harness.NewList(QueueKind::Graphics, list));
            const BufferBarrier upload[] = {
                {staging.Value(), ResourceState::Undefined, ResourceState::CopySource},
                {stream.Value(), streamState, ResourceState::CopyDestination},
                {arguments.Value(), argumentState, ResourceState::CopyDestination},
            };
            CG_REQUIRE(list.Barrier(Span<const BufferBarrier>(upload, 3)).IsOk());
            CG_REQUIRE(list.CopyBuffer(BufferCopy{staging.Value(), 0, stream.Value(), 0,
                                                  kIds * sizeof(u32)})
                           .IsOk());
            CG_REQUIRE(list.CopyBuffer(BufferCopy{staging.Value(), argumentsAt, arguments.Value(),
                                                  0, sizeof(argumentBytes)})
                           .IsOk());
            const BufferBarrier ready[] = {
                {stream.Value(), ResourceState::CopyDestination, ResourceState::VertexBuffer},
                {arguments.Value(), ResourceState::CopyDestination,
                 ResourceState::IndirectArgument},
            };
            CG_REQUIRE(list.Barrier(Span<const BufferBarrier>(ready, 2)).IsOk());
            streamState = ResourceState::VertexBuffer;
            argumentState = ResourceState::IndirectArgument;
            CG_REQUIRE(BeginPointPass(target, list));
            CG_REQUIRE(
                list.SetVertexBuffer(kGpuDrivenInstanceStreamBinding, stream.Value()).IsOk());
            IndirectCountArguments countArguments{};
            countArguments.argumentBuffer = arguments.Value();
            countArguments.countBuffer = arguments.Value();
            countArguments.countOffsetBytes = 128;
            countArguments.maxDrawCount = c.maxDraws;
            const u32 drawsBefore = list.Stats().draws;
            CG_REQUIRE(list.DrawIndexedIndirectCount(countArguments).IsOk());
            CG_REQUIRE(list.DrawIndirect(arguments.Value(), 160, 2, 16).IsOk());
            CG_EXPECT_EQ(list.Stats().draws, drawsBefore + 2);
            CG_REQUIRE(EndPointPass(target, list, true));
            CG_REQUIRE(harness.Run(list));

            DynamicArray<u8> lit(heap);
            CG_REQUIRE(ReadLitIds(device, target, heap, lit));
            u32 wrong = 0;
            for (u32 id = 0; id < kIds; ++id)
            {
                bool expected = false;
                for (u32 draw = 0; draw < c.expectedDraws; ++draw)
                {
                    expected =
                        expected || (id >= draws[draw].firstInstance &&
                                     id < draws[draw].firstInstance + draws[draw].instanceCount);
                }
                for (const DrawArguments& plain : plainDraws)
                {
                    expected = expected || (id >= plain.firstInstance &&
                                            id < plain.firstInstance + plain.instanceCount);
                }
                wrong += (lit[id] != 0) == expected ? 0U : 1U;
            }
            if (wrong != 0)
            {
                u32 litCount = 0;
                u32 firstLit = kIds;
                for (u32 id = 0; id < lit.Size(); ++id)
                {
                    litCount += lit[id] != 0 ? 1U : 0U;
                    firstLit = (lit[id] != 0 && firstLit == kIds) ? id : firstLit;
                }
                std::printf("adapter \"%s\": count %u max %u: %u ids wrong, %u texels lit, "
                            "first at %u\n",
                            adapters[adapterIndex].name, c.gpuCount, c.maxDraws, wrong, litCount,
                            firstLit);
            }
            CG_EXPECT_EQ(wrong, u32{0});
        }
        CG_REQUIRE(device.WaitIdle().IsOk());
        (void)device.DestroyBuffer(stream.Value());
        (void)device.DestroyBuffer(arguments.Value());
        (void)device.DestroyBuffer(staging.Value());
    }
    (void)cyber::core::DefaultLogger().RemoveSink(&sink);
    if (watch.records != 0)
    {
        CG_FAIL(watch.last);
    }
}

// ---------------------------------------------------------------------------
// GPU culling against the CPU
// ---------------------------------------------------------------------------

// The roadmap's acceptance scene: a million instances culled on the GPU,
// compared instance by instance with the CPU reference and in total with
// Phase 22's `Scene::Cull`, on every adapter.
CG_TEST(gpu_driven_d3d12_culls_a_million_instances_like_the_cpu)
{
    Adapter adapters[8]{};
    const u32 count = ListAdapters(adapters);
    if (count == 0)
    {
        CG_SKIP("no Direct3D 12 adapter");
    }
    HeapAllocator heap;
    World world(heap);
    CG_REQUIRE(BuildWorld(world, kMillion, 41));
    const GpuCullView view = DefaultView();
    GpuCullOutput reference(heap);
    CG_REQUIRE(CullInstancesReference(world.scene, view, nullptr, reference).IsOk());
    DynamicArray<u32> referenceLods(heap);
    CG_REQUIRE(LodsById(world.scene, reference, referenceLods));

    // Phase 22's culler over the same million, for the visible total.
    {
        auto jobsResult = cyber::jobs::JobSystem::Create(cyber::jobs::JobSystem::Desc{4}, heap);
        CG_REQUIRE(static_cast<bool>(jobsResult));
        cyber::jobs::JobSystem jobs = std::move(jobsResult.Value());
        constexpr u32 kPacket = 50000;
        auto writerResult = cyber::scene::SceneCommandWriter::Create({kMillion, kPacket}, heap);
        auto sceneResult = cyber::scene::Scene::Create({kMillion, kPacket}, heap);
        CG_REQUIRE(static_cast<bool>(writerResult) && static_cast<bool>(sceneResult));
        cyber::scene::SceneCommandWriter writer = std::move(writerResult.Value());
        cyber::scene::Scene phase22 = std::move(sceneResult.Value());
        DynamicArray<u32> instanceOfSlot(heap);
        CG_REQUIRE(instanceOfSlot.TryResize(kMillion));
        for (u32 index = 0; index < kMillion; ++index)
        {
            const GpuDrawGroup& group = world.scene.Groups()[index % kGroups];
            cyber::scene::MeshInstanceDesc desc{};
            desc.meshId = 1 + index % kGroups;
            desc.materialId = 1;
            desc.transform = Mat4::Translation(world.translations[index]);
            desc.localBounds = Aabb::FromCenterExtents(Vec3::Zero(), world.extents[index]);
            desc.lodCount = group.lodCount;
            for (u32 lod = 0; lod + 1 < group.lodCount; ++lod)
            {
                desc.lodPixelThresholds[lod] = group.lodPixelThresholds[lod];
            }
            auto handle = writer.CreateMeshInstance(desc);
            CG_REQUIRE(static_cast<bool>(handle) && handle.Value().index < kMillion);
            instanceOfSlot[handle.Value().index] = index;
            if ((index + 1) % kPacket == 0)
            {
                CG_REQUIRE(phase22.ApplyCommands(writer.Seal()).IsOk());
                CG_REQUIRE(writer.Commit().IsOk());
            }
        }
        cyber::scene::SceneView sceneView{};
        sceneView.frustum = view.frustum;
        sceneView.cameraPosition = view.cameraPosition;
        sceneView.projectionScalePixels = view.projectionScalePixels;
        sceneView.nearDistance = view.nearDistance;
        auto visible = phase22.Cull(sceneView, jobs, heap);
        CG_REQUIRE(static_cast<bool>(visible));
        u32 lodsMatching = 0;
        const auto entries = visible.Value().Entries();
        for (usize entry = 0; entry < entries.Size(); ++entry)
        {
            lodsMatching +=
                referenceLods[instanceOfSlot[entries[entry].handle.index]] == entries[entry].lod
                    ? 1U
                    : 0U;
        }
        CG_EXPECT_EQ(static_cast<u32>(entries.Size()), reference.counters.visible);
        CG_EXPECT_EQ(lodsMatching, reference.counters.visible);
    }

    World softwareWorld(heap);
    GpuCullOutput softwareReference(heap);
    DynamicArray<u32> softwareLods(heap);
    for (u32 adapterIndex = 0; adapterIndex < count; ++adapterIndex)
    {
        const u32 instances = InstancesFor(adapters[adapterIndex], kMillion);
        if (instances != kMillion && softwareWorld.scene.Instances().Empty())
        {
            CG_REQUIRE(BuildWorld(softwareWorld, instances, 41));
            CG_REQUIRE(CullInstancesReference(softwareWorld.scene, view, nullptr, softwareReference)
                           .IsOk());
            CG_REQUIRE(LodsById(softwareWorld.scene, softwareReference, softwareLods));
        }
        const World& tested = instances == kMillion ? world : softwareWorld;
        GpuCullOutput& expected = instances == kMillion ? reference : softwareReference;
        const DynamicArray<u32>& expectedLods =
            instances == kMillion ? referenceLods : softwareLods;

        Harness harness(heap);
        CG_REQUIRE(harness.Open(adapters[adapterIndex]));
        auto created =
            GpuCullingPipeline::Create(harness.device, PipelineDescFor(tested.scene), heap);
        CG_REQUIRE(static_cast<bool>(created));
        GpuCullingPipeline pipeline = std::move(created.Value());

        CommandList upload;
        CG_REQUIRE(harness.NewList(QueueKind::Graphics, upload));
        CG_REQUIRE(pipeline.RecordSceneUpload(upload, tested.scene).IsOk());
        CG_REQUIRE(harness.Run(upload));

        CommandList cull;
        CG_REQUIRE(harness.NewList(QueueKind::Graphics, cull));
        CG_REQUIRE(pipeline.RecordCull(cull, view).IsOk());
        const u32 cullCommands = cull.Stats().dispatches;
        CG_REQUIRE(pipeline.RecordReadback(cull).IsOk());
        const auto start = std::chrono::steady_clock::now();
        CG_REQUIRE(harness.Run(cull));
        const double milliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();

        GpuCullOutput gpu(heap);
        CG_REQUIRE(pipeline.ReadBack(gpu).IsOk());
        DynamicArray<u32> gpuLods(heap);
        CG_REQUIRE(LodsById(tested.scene, gpu, gpuLods));
        const Agreement agreement =
            CompareDecisions(tested.scene, view, nullptr, gpuLods, expectedLods);
        std::printf("adapter \"%s\": %u of %u visible on the GPU (CPU %u), %u draws, "
                    "%u dispatches, %.2f ms submit-to-idle; %u disagreements (%u unexplained)\n",
                    adapters[adapterIndex].name, gpu.counters.visible, instances,
                    expected.counters.visible, gpu.counters.drawCount, cullCommands, milliseconds,
                    agreement.mismatches, agreement.unexplained);
        CG_EXPECT_EQ(agreement.unexplained, u32{0});
        CG_EXPECT_EQ(gpu.counters.overflow, u32{0});
        CG_EXPECT_EQ(gpu.counters.visible + gpu.counters.frustumCulled, instances);
        // Two dispatches for a million instances.
        CG_EXPECT_EQ(cullCommands, u32{2});
        if (agreement.mismatches == 0)
        {
            CG_EXPECT_EQ(gpu.counters.visible, expected.counters.visible);
            CG_EXPECT_EQ(gpu.counters.drawCount, expected.counters.drawCount);
            CG_EXPECT_TRUE(SameVisibleSets(tested.scene, gpu, expected));
            // Compaction is ordered, so the draws match byte for byte.
            bool drawsIdentical = gpu.draws.Size() == expected.draws.Size();
            for (usize draw = 0; drawsIdentical && draw < gpu.draws.Size(); ++draw)
            {
                drawsIdentical = std::memcmp(&gpu.draws[draw], &expected.draws[draw],
                                             sizeof(DrawIndexedArguments)) == 0;
            }
            CG_EXPECT_TRUE(drawsIdentical);
        }
        CG_REQUIRE(harness.device.WaitIdle().IsOk());
    }
}

// Culled on the GPU, drawn with one indirect call, and every drawn instance
// is exactly one the culler kept: a point per surviving identifier, read back.
CG_TEST(gpu_driven_d3d12_draws_a_culled_million_with_one_call)
{
    Adapter adapters[8]{};
    const u32 count = ListAdapters(adapters);
    if (count == 0)
    {
        CG_SKIP("no Direct3D 12 adapter");
    }
    HeapAllocator heap;
    DebugLayerWatch watch{};
    cyber::core::CallbackSink sink{&WatchLog, &watch};
    CG_REQUIRE(cyber::core::DefaultLogger().AddSink(&sink));
    {
        const GpuCullView view = DefaultView();
        for (u32 adapterIndex = 0; adapterIndex < count; ++adapterIndex)
        {
            const u32 instances = InstancesFor(adapters[adapterIndex], kMillion);
            World world(heap);
            CG_REQUIRE(BuildWorld(world, instances, 43));
            Harness harness(heap);
            CG_REQUIRE(harness.Open(adapters[adapterIndex]));
            PointTarget target;
            CG_REQUIRE(BuildPointTarget(harness.device, target));
            auto created =
                GpuCullingPipeline::Create(harness.device, PipelineDescFor(world.scene), heap);
            CG_REQUIRE(static_cast<bool>(created));
            GpuCullingPipeline pipeline = std::move(created.Value());

            CommandList frame;
            CG_REQUIRE(harness.NewList(QueueKind::Graphics, frame));
            CG_REQUIRE(pipeline.RecordSceneUpload(frame, world.scene).IsOk());
            CG_REQUIRE(pipeline.RecordCull(frame, view).IsOk());
            CG_REQUIRE(pipeline.RecordAcquireForDraw(frame).IsOk());
            CG_REQUIRE(BeginPointPass(target, frame));
            const u32 drawsBefore = frame.Stats().draws;
            CG_REQUIRE(pipeline.RecordDraw(frame).IsOk());
            const u32 drawCalls = frame.Stats().draws - drawsBefore;
            CG_REQUIRE(EndPointPass(target, frame, true));
            CG_REQUIRE(pipeline.RecordReadback(frame).IsOk());
            CG_REQUIRE(harness.Run(frame));

            GpuCullOutput gpu(heap);
            CG_REQUIRE(pipeline.ReadBack(gpu).IsOk());
            DynamicArray<u32> gpuLods(heap);
            CG_REQUIRE(LodsById(world.scene, gpu, gpuLods));
            DynamicArray<u8> lit(heap);
            CG_REQUIRE(ReadLitIds(harness.device, target, heap, lit));
            u32 wrong = 0;
            u32 drawn = 0;
            for (u32 id = 0; id < instances; ++id)
            {
                const bool kept = gpuLods[id] != kHidden;
                wrong += (lit[id] != 0) == kept ? 0U : 1U;
                drawn += lit[id] != 0 ? 1U : 0U;
            }
            std::printf("adapter \"%s\": %u instances drawn by %u draw call(s) carrying %u "
                        "indirect draws\n",
                        adapters[adapterIndex].name, drawn, drawCalls, gpu.counters.drawCount);
            CG_EXPECT_EQ(wrong, u32{0});
            CG_EXPECT_EQ(drawn, gpu.counters.visible);
            CG_EXPECT_EQ(drawCalls, u32{1});
            CG_EXPECT_EQ(pipeline.Stats().drawCalls, u32{1});
            CG_EXPECT_TRUE(gpu.counters.drawCount <= world.scene.Batches().Size());
            CG_REQUIRE(harness.device.WaitIdle().IsOk());
        }
    }
    (void)cyber::core::DefaultLogger().RemoveSink(&sink);
    if (watch.records != 0)
    {
        CG_FAIL(watch.last);
    }
}

// ---------------------------------------------------------------------------
// Hi-Z on the GPU
// ---------------------------------------------------------------------------

namespace
{

// Reverse-Z depth of a wall of the given half-size at `distance`, everything
// else at the far plane, plus a ramp so no two texels of the pyramid agree by
// accident.
[[nodiscard]] bool WallDepth(u32 width, u32 height, u32 pitch, f32 halfWidth, f32 halfHeight,
                             f32 distance, DynamicArray<f32>& depth) noexcept
{
    if (!depth.TryResize(static_cast<usize>(pitch) * height))
    {
        return false;
    }
    const Mat4 projection = Mat4::PerspectiveReverseZ(
        kFov, static_cast<f32>(kViewportWidth) / static_cast<f32>(kViewportHeight), kNear, kFar);
    const Vec4 clip = projection * Vec4{0.0f, 0.0f, -distance, 1.0f};
    const f32 wall = clip.z / clip.w;
    const f32 aspect = static_cast<f32>(kViewportWidth) / static_cast<f32>(kViewportHeight);
    const f32 tanHalf = std::tan(kFov * 0.5f);
    for (u32 y = 0; y < height; ++y)
    {
        for (u32 x = 0; x < pitch; ++x)
        {
            const f32 ndcX = ((static_cast<f32>(x) + 0.5f) / static_cast<f32>(width)) * 2.0f - 1.0f;
            const f32 ndcY =
                1.0f - ((static_cast<f32>(y) + 0.5f) / static_cast<f32>(height)) * 2.0f;
            const bool inside = std::fabs(ndcX * tanHalf * aspect * distance) <= halfWidth &&
                                std::fabs(ndcY * tanHalf * distance) <= halfHeight;
            const f32 ramp = 1.0e-6f * static_cast<f32>((x * 7 + y * 13) % 97);
            depth[static_cast<usize>(y) * pitch + x] = inside ? wall - ramp : ramp;
        }
    }
    return true;
}

}  // namespace

// The GPU pyramid is the CPU pyramid, texel for texel: a reduction of minima
// has no rounding to differ by.
CG_TEST(gpu_driven_d3d12_hiz_build_matches_the_cpu_pyramid)
{
    Adapter adapters[8]{};
    const u32 count = ListAdapters(adapters);
    if (count == 0)
    {
        CG_SKIP("no Direct3D 12 adapter");
    }
    HeapAllocator heap;
    constexpr u32 kWidth = 157;
    constexpr u32 kHeight = 93;
    constexpr u32 kPitch = 160;
    DynamicArray<f32> depth(heap);
    CG_REQUIRE(WallDepth(kWidth, kHeight, kPitch, 20.0f, 12.0f, 40.0f, depth));
    HiZPyramid cpu(heap);
    CG_REQUIRE(
        cpu.Build(Span<const f32>(depth.Data(), depth.Size()), kWidth, kHeight, kPitch).IsOk());

    World world(heap);
    CG_REQUIRE(BuildWorld(world, 64, 1));
    for (u32 adapterIndex = 0; adapterIndex < count; ++adapterIndex)
    {
        Harness harness(heap);
        CG_REQUIRE(harness.Open(adapters[adapterIndex]));
        Device& device = harness.device;
        auto created =
            GpuCullingPipeline::Create(device, PipelineDescFor(world.scene, kWidth, kHeight), heap);
        CG_REQUIRE(static_cast<bool>(created));
        GpuCullingPipeline pipeline = std::move(created.Value());

        BufferDesc depthDesc{};
        depthDesc.usage = BufferUsage::Storage | BufferUsage::CopyDestination;
        depthDesc.sizeBytes = depth.Size() * sizeof(f32);
        depthDesc.elementStride = sizeof(f32);
        BufferDesc stagingDesc{};
        stagingDesc.usage = BufferUsage::HostVisible | BufferUsage::CopySource;
        stagingDesc.sizeBytes = depth.Size() * sizeof(f32);
        auto depthBuffer = device.CreateBuffer(depthDesc);
        auto staging = device.CreateBuffer(stagingDesc);
        CG_REQUIRE(static_cast<bool>(depthBuffer) && static_cast<bool>(staging));
        CG_REQUIRE(device
                       .WriteBuffer(staging.Value(), 0,
                                    Span<const Byte>(reinterpret_cast<const Byte*>(depth.Data()),
                                                     depth.Size() * sizeof(f32)))
                       .IsOk());

        CommandList list;
        CG_REQUIRE(harness.NewList(QueueKind::Graphics, list));
        const BufferBarrier upload[] = {
            {staging.Value(), ResourceState::Undefined, ResourceState::CopySource},
            {depthBuffer.Value(), ResourceState::Undefined, ResourceState::CopyDestination},
        };
        CG_REQUIRE(list.Barrier(Span<const BufferBarrier>(upload, 2)).IsOk());
        CG_REQUIRE(list.CopyBuffer(BufferCopy{staging.Value(), 0, depthBuffer.Value(), 0,
                                              depth.Size() * sizeof(f32)})
                       .IsOk());
        const BufferBarrier read{depthBuffer.Value(), ResourceState::CopyDestination,
                                 ResourceState::ShaderRead};
        CG_REQUIRE(list.Barrier(Span<const BufferBarrier>(&read, 1)).IsOk());
        CG_REQUIRE(pipeline.RecordSceneUpload(list, world.scene).IsOk());
        CG_EXPECT_FALSE(pipeline.RecordHiZBuild(list, depthBuffer.Value(), kWidth + 1, kHeight));
        CG_REQUIRE(
            pipeline.RecordHiZBuild(list, depthBuffer.Value(), kWidth, kHeight, kPitch).IsOk());
        const u32 reductions = list.Stats().dispatches;
        CG_REQUIRE(pipeline.RecordReadback(list).IsOk());
        CG_REQUIRE(harness.Run(list));

        const Span<const f32> expected = cpu.Texels();
        DynamicArray<f32> texels(heap);
        CG_REQUIRE(texels.TryResize(expected.Size()));
        CG_REQUIRE(pipeline.ReadBackHiZ(Span<f32>(texels.Data(), texels.Size())).IsOk());
        u32 different = 0;
        for (usize index = 0; index < expected.Size(); ++index)
        {
            different += std::memcmp(&texels[index], &expected[index], sizeof(f32)) == 0 ? 0U : 1U;
        }
        std::printf("adapter \"%s\": %zu-texel pyramid in %u dispatches, %u texels differ\n",
                    adapters[adapterIndex].name, expected.Size(), reductions, different);
        CG_EXPECT_EQ(different, u32{0});
        CG_EXPECT_EQ(reductions, cpu.LevelCount());
        CG_REQUIRE(device.WaitIdle().IsOk());
        (void)device.DestroyBuffer(depthBuffer.Value());
        (void)device.DestroyBuffer(staging.Value());
    }
}

// Occlusion on the GPU against the CPU reference, through a pyramid the GPU
// built itself from last frame's depth.
CG_TEST(gpu_driven_d3d12_occlusion_culling_matches_the_reference)
{
    Adapter adapters[8]{};
    const u32 count = ListAdapters(adapters);
    if (count == 0)
    {
        CG_SKIP("no Direct3D 12 adapter");
    }
    HeapAllocator heap;
    constexpr u32 kWidth = 320;
    constexpr u32 kHeight = 180;
    // A wall nearly filling the view at distance 150: most of the world
    // behind it is hidden, everything in front and around it is not.
    DynamicArray<f32> depth(heap);
    CG_REQUIRE(WallDepth(kWidth, kHeight, kWidth, 110.0f, 60.0f, 150.0f, depth));
    HiZPyramid cpu(heap);
    CG_REQUIRE(cpu.Build(Span<const f32>(depth.Data(), depth.Size()), kWidth, kHeight).IsOk());

    const Vec3 eye{0.0f, 0.0f, 0.0f};
    GpuCullView view = MakeGpuCullView(ViewProjection(eye, Vec3{0.0f, 0.0f, -1.0f}), eye, kFov,
                                       kViewportHeight, kNear);
    view.flags = kGpuCullAll;

    for (u32 adapterIndex = 0; adapterIndex < count; ++adapterIndex)
    {
        World world(heap);
        CG_REQUIRE(BuildWorld(world, InstancesFor(adapters[adapterIndex], 200000), 47));
        GpuCullOutput reference(heap);
        CG_REQUIRE(CullInstancesReference(world.scene, view, &cpu, reference).IsOk());
        DynamicArray<u32> referenceLods(heap);
        CG_REQUIRE(LodsById(world.scene, reference, referenceLods));
        CG_EXPECT_TRUE(reference.counters.occlusionCulled > reference.counters.visible / 4);

        Harness harness(heap);
        CG_REQUIRE(harness.Open(adapters[adapterIndex]));
        Device& device = harness.device;
        auto created =
            GpuCullingPipeline::Create(device, PipelineDescFor(world.scene, kWidth, kHeight), heap);
        CG_REQUIRE(static_cast<bool>(created));
        GpuCullingPipeline pipeline = std::move(created.Value());
        BufferDesc depthDesc{};
        depthDesc.usage = BufferUsage::Storage | BufferUsage::CopyDestination;
        depthDesc.sizeBytes = depth.Size() * sizeof(f32);
        depthDesc.elementStride = sizeof(f32);
        BufferDesc stagingDesc{};
        stagingDesc.usage = BufferUsage::HostVisible | BufferUsage::CopySource;
        stagingDesc.sizeBytes = depthDesc.sizeBytes;
        auto depthBuffer = device.CreateBuffer(depthDesc);
        auto staging = device.CreateBuffer(stagingDesc);
        CG_REQUIRE(static_cast<bool>(depthBuffer) && static_cast<bool>(staging));
        CG_REQUIRE(device
                       .WriteBuffer(staging.Value(), 0,
                                    Span<const Byte>(reinterpret_cast<const Byte*>(depth.Data()),
                                                     depth.Size() * sizeof(f32)))
                       .IsOk());

        CommandList list;
        CG_REQUIRE(harness.NewList(QueueKind::Graphics, list));
        const BufferBarrier upload[] = {
            {staging.Value(), ResourceState::Undefined, ResourceState::CopySource},
            {depthBuffer.Value(), ResourceState::Undefined, ResourceState::CopyDestination},
        };
        CG_REQUIRE(list.Barrier(Span<const BufferBarrier>(upload, 2)).IsOk());
        CG_REQUIRE(list.CopyBuffer(BufferCopy{staging.Value(), 0, depthBuffer.Value(), 0,
                                              depthDesc.sizeBytes})
                       .IsOk());
        const BufferBarrier read{depthBuffer.Value(), ResourceState::CopyDestination,
                                 ResourceState::ShaderRead};
        CG_REQUIRE(list.Barrier(Span<const BufferBarrier>(&read, 1)).IsOk());
        CG_REQUIRE(pipeline.RecordSceneUpload(list, world.scene).IsOk());
        CG_REQUIRE(pipeline.RecordHiZBuild(list, depthBuffer.Value(), kWidth, kHeight).IsOk());
        CG_REQUIRE(pipeline.RecordCull(list, view).IsOk());
        CG_REQUIRE(pipeline.RecordReadback(list).IsOk());
        CG_REQUIRE(harness.Run(list));

        GpuCullOutput gpu(heap);
        CG_REQUIRE(pipeline.ReadBack(gpu).IsOk());
        DynamicArray<u32> gpuLods(heap);
        CG_REQUIRE(LodsById(world.scene, gpu, gpuLods));
        const Agreement agreement =
            CompareDecisions(world.scene, view, &cpu, gpuLods, referenceLods);
        std::printf("adapter \"%s\": occlusion culled %u (CPU %u), visible %u (CPU %u); "
                    "%u disagreements (%u unexplained)\n",
                    adapters[adapterIndex].name, gpu.counters.occlusionCulled,
                    reference.counters.occlusionCulled, gpu.counters.visible,
                    reference.counters.visible, agreement.mismatches, agreement.unexplained);
        CG_EXPECT_EQ(agreement.unexplained, u32{0});
        CG_EXPECT_EQ(gpu.counters.visible + gpu.counters.frustumCulled +
                         gpu.counters.occlusionCulled,
                     static_cast<u32>(world.scene.Instances().Size()));
        CG_REQUIRE(device.WaitIdle().IsOk());
        (void)device.DestroyBuffer(depthBuffer.Value());
        (void)device.DestroyBuffer(staging.Value());
    }
}

// ---------------------------------------------------------------------------
// Async compute
// ---------------------------------------------------------------------------

namespace
{

// One frame: culling recorded for `cullQueue`, drawing on the graphics queue.
// With the compute queue, the graphics submission waits on a fence the compute
// submission signals; on one queue the two lists run back to back.
[[nodiscard]] bool RunFrame(Harness& harness, GpuCullingPipeline& pipeline, PointTarget& target,
                            const GpuCullView& view, QueueKind cullQueue, bool readback,
                            double* milliseconds) noexcept
{
    CommandList cull;
    CommandList draw;
    if (!harness.NewList(cullQueue, cull))
    {
        std::printf("  frame: the %s list could not be created\n", QueueKindName(cullQueue));
        return false;
    }
    if (const Status<cyber::core::Error> culled = pipeline.RecordCull(cull, view); !culled)
    {
        std::printf("  frame: RecordCull on the %s queue failed: %s\n", QueueKindName(cullQueue),
                    culled.Error().message);
        return false;
    }
    if (!cull.End() || !harness.NewList(QueueKind::Graphics, draw) ||
        !pipeline.RecordAcquireForDraw(draw) || !BeginPointPass(target, draw) ||
        !pipeline.RecordDraw(draw) || !EndPointPass(target, draw, readback) ||
        (readback && !pipeline.RecordReadback(draw)) || !draw.End())
    {
        return false;
    }
    const auto start = std::chrono::steady_clock::now();
    const u64 culled = ++harness.fenceValue;
    const CommandListHandle cullHandle = cull.Handle();
    const FenceSignal cullSignal{harness.fence, culled};
    SubmitDesc cullSubmit{};
    cullSubmit.queue = cullQueue;
    cullSubmit.commandLists = Span<const CommandListHandle>(&cullHandle, 1);
    cullSubmit.signals = Span<const FenceSignal>(&cullSignal, 1);
    cullSubmit.label = "gpu culling";
    if (const auto submitted = harness.device.Submit(cullSubmit); !submitted)
    {
        std::printf("  frame: the culling submission failed: %s\n", submitted.Error().message);
        return false;
    }
    const u64 drawn = ++harness.fenceValue;
    const CommandListHandle drawHandle = draw.Handle();
    const FenceSignal drawSignal{harness.fence, drawn};
    // The draw must not read the arguments before the culling writes them.
    const GpuWait wait{harness.fence, culled};
    SubmitDesc drawSubmit{};
    drawSubmit.queue = QueueKind::Graphics;
    drawSubmit.commandLists = Span<const CommandListHandle>(&drawHandle, 1);
    drawSubmit.signals = Span<const FenceSignal>(&drawSignal, 1);
    drawSubmit.gpuWaits = Span<const GpuWait>(&wait, cullQueue == QueueKind::Graphics ? 0 : 1);
    drawSubmit.label = "gpu-driven draw";
    if (const auto submitted = harness.device.Submit(drawSubmit); !submitted)
    {
        std::printf("  frame: the draw submission failed: %s\n", submitted.Error().message);
        return false;
    }
    if (!harness.device.WaitFence(harness.fence, drawn))
    {
        return false;
    }
    if (milliseconds != nullptr)
    {
        *milliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
    }
    return true;
}

[[nodiscard]] double Median(double* values, u32 count) noexcept
{
    std::sort(values, values + count);
    return count % 2 == 1 ? values[count / 2] : 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

// The fastest of several frames. Preemption and other processes only ever add
// time to a frame, so under load (ctest -j, a busy desktop) the median of a
// short measurement drifts up while the minimum stays at the intrinsic cost.
// Ratios between sizes are taken between minima for that reason.
[[nodiscard]] double Fastest(const double* values, u32 count) noexcept
{
    return *std::min_element(values, values + count);
}

}  // namespace

// Culling on the compute queue, drawing on the graphics queue, ordered by a
// fence: the image is the one the single-queue frame produces, the debug
// layer is silent about the cross-queue hand-off, and the measured timings
// decide whether `Auto` would keep async compute on this adapter.
CG_TEST(gpu_driven_d3d12_async_compute_culling_matches_the_graphics_queue)
{
    Adapter adapters[8]{};
    const u32 count = ListAdapters(adapters);
    if (count == 0)
    {
        CG_SKIP("no Direct3D 12 adapter");
    }
    HeapAllocator heap;
    DebugLayerWatch watch{};
    cyber::core::CallbackSink sink{&WatchLog, &watch};
    CG_REQUIRE(cyber::core::DefaultLogger().AddSink(&sink));
    {
        const GpuCullView view = DefaultView();
        for (u32 adapterIndex = 0; adapterIndex < count; ++adapterIndex)
        {
            World world(heap);
            CG_REQUIRE(BuildWorld(world, InstancesFor(adapters[adapterIndex], kMillion), 53));
            Harness harness(heap);
            CG_REQUIRE(harness.Open(adapters[adapterIndex]));
            if (!harness.device.Queue(QueueKind::Compute).IsValid())
            {
                std::printf("adapter \"%s\": no compute queue\n", adapters[adapterIndex].name);
                continue;
            }
            PointTarget target;
            CG_REQUIRE(BuildPointTarget(harness.device, target));
            auto created =
                GpuCullingPipeline::Create(harness.device, PipelineDescFor(world.scene), heap);
            CG_REQUIRE(static_cast<bool>(created));
            GpuCullingPipeline pipeline = std::move(created.Value());
            CommandList upload;
            CG_REQUIRE(harness.NewList(QueueKind::Graphics, upload));
            CG_REQUIRE(pipeline.RecordSceneUpload(upload, world.scene).IsOk());
            CG_REQUIRE(harness.Run(upload));

            CG_REQUIRE(
                RunFrame(harness, pipeline, target, view, QueueKind::Graphics, true, nullptr));
            DynamicArray<u8> serialLit(heap);
            CG_REQUIRE(ReadLitIds(harness.device, target, heap, serialLit));
            GpuCullOutput serial(heap);
            CG_REQUIRE(pipeline.ReadBack(serial).IsOk());

            CG_REQUIRE(
                RunFrame(harness, pipeline, target, view, QueueKind::Compute, true, nullptr));
            DynamicArray<u8> asyncLit(heap);
            CG_REQUIRE(ReadLitIds(harness.device, target, heap, asyncLit));
            GpuCullOutput async(heap);
            CG_REQUIRE(pipeline.ReadBack(async).IsOk());
            u32 different = 0;
            u32 drawn = 0;
            for (usize id = 0; id < serialLit.Size(); ++id)
            {
                different += serialLit[id] == asyncLit[id] ? 0U : 1U;
                drawn += asyncLit[id];
            }
            CG_EXPECT_EQ(different, u32{0});
            CG_EXPECT_EQ(drawn, async.counters.visible);
            CG_EXPECT_EQ(async.counters.visible, serial.counters.visible);
            CG_EXPECT_EQ(async.counters.drawCount, serial.counters.drawCount);

            // Measure before keeping it: medians of both schedules.
            constexpr u32 kFrames = 12;
            double serialTimes[kFrames]{};
            double asyncTimes[kFrames]{};
            for (u32 frame = 0; frame < kFrames; ++frame)
            {
                CG_REQUIRE(RunFrame(harness, pipeline, target, view, QueueKind::Graphics, false,
                                    &serialTimes[frame]));
                CG_REQUIRE(RunFrame(harness, pipeline, target, view, QueueKind::Compute, false,
                                    &asyncTimes[frame]));
            }
            AsyncComputeMeasurement measurement{};
            measurement.serialMilliseconds = static_cast<f32>(Median(serialTimes, kFrames));
            measurement.asyncMilliseconds = static_cast<f32>(Median(asyncTimes, kFrames));
            measurement.samples = kFrames;
            const AsyncComputePlan plan =
                SelectAsyncComputePlan(AsyncComputePolicyDesc{}, true, &measurement);
            std::printf("adapter \"%s\": frame %.3f ms on one queue, %.3f ms with async compute "
                        "(gain %+.1f%%); Auto %s async compute (%s)\n",
                        adapters[adapterIndex].name, measurement.serialMilliseconds,
                        measurement.asyncMilliseconds, plan.measuredGainFraction * 100.0f,
                        plan.useAsyncCompute ? "keeps" : "does not keep",
                        AsyncComputeReasonName(plan.reason));
            CG_EXPECT_EQ(plan.useAsyncCompute, plan.measuredGainFraction >= 0.05f);
            CG_REQUIRE(harness.device.WaitIdle().IsOk());
        }
    }
    (void)cyber::core::DefaultLogger().RemoveSink(&sink);
    if (watch.records != 0)
    {
        CG_FAIL(watch.last);
    }
}

// ---------------------------------------------------------------------------
// CPU time against object count
// ---------------------------------------------------------------------------

// The roadmap's "CPU time stays flat as object count grows": the render
// thread's cost of a GPU-driven frame — writing the cull constants and
// recording and submitting the culling and the indirect draw — is measured
// for a thousand to a million instances, with the GPU's own time excluded by
// stopping the clock at submission. A CPU-driven control (the Phase 22 cull
// plus one draw per visible instance) is measured beside it to show the bar
// can fail.
CG_TEST(gpu_driven_d3d12_cpu_time_stays_flat_as_objects_grow)
{
    Adapter adapters[8]{};
    const u32 count = ListAdapters(adapters);
    if (count == 0)
    {
        CG_SKIP("no Direct3D 12 adapter");
    }
    // The first adapter is hardware when there is any.
    const Adapter& adapter = adapters[0];
    HeapAllocator heap;
    Harness harness(heap);
    CG_REQUIRE(harness.Open(adapter));
    PointTarget target;
    CG_REQUIRE(BuildPointTarget(harness.device, target));

    constexpr u32 kSizes[] = {1000, 10000, 100000, kMillion};
    constexpr u32 kSizeCount = sizeof(kSizes) / sizeof(kSizes[0]);
    constexpr u32 kFrames = 25;
    double gpuDriven[kSizeCount]{};
    double cpuDriven[kSizeCount]{};
    for (u32 sizeIndex = 0; sizeIndex < kSizeCount; ++sizeIndex)
    {
        World world(heap);
        CG_REQUIRE(BuildWorld(world, kSizes[sizeIndex], 59));
        const GpuCullView view = DefaultView();
        auto created =
            GpuCullingPipeline::Create(harness.device, PipelineDescFor(world.scene), heap);
        CG_REQUIRE(static_cast<bool>(created));
        GpuCullingPipeline pipeline = std::move(created.Value());
        CommandList upload;
        CG_REQUIRE(harness.NewList(QueueKind::Graphics, upload));
        CG_REQUIRE(pipeline.RecordSceneUpload(upload, world.scene).IsOk());
        CG_REQUIRE(harness.Run(upload));

        double times[kFrames]{};
        for (u32 frame = 0; frame < kFrames; ++frame)
        {
            CommandList list;
            CG_REQUIRE(harness.NewList(QueueKind::Graphics, list));
            const auto start = std::chrono::steady_clock::now();
            CG_REQUIRE(pipeline.RecordCull(list, view).IsOk());
            CG_REQUIRE(pipeline.RecordAcquireForDraw(list).IsOk());
            CG_REQUIRE(BeginPointPass(target, list));
            CG_REQUIRE(pipeline.RecordDraw(list).IsOk());
            CG_REQUIRE(EndPointPass(target, list, false));
            CG_REQUIRE(list.End().IsOk());
            const CommandListHandle handle = list.Handle();
            const u64 value = ++harness.fenceValue;
            const FenceSignal signal{harness.fence, value};
            SubmitDesc submit{};
            submit.queue = QueueKind::Graphics;
            submit.commandLists = Span<const CommandListHandle>(&handle, 1);
            submit.signals = Span<const FenceSignal>(&signal, 1);
            CG_REQUIRE(harness.device.Submit(submit).IsOk());
            times[frame] =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            CG_REQUIRE(harness.device.WaitFence(harness.fence, value).IsOk());
        }
        gpuDriven[sizeIndex] = Fastest(times, kFrames);

        // The control stops at 100,000 instances: a million CPU-recorded draws
        // a frame says nothing the trend does not.
        if (kSizes[sizeIndex] <= 100000)
        {
            constexpr u32 kControlFrames = 9;
            double control[kControlFrames]{};
            GpuCullOutput culled(heap);
            for (u32 frame = 0; frame < kControlFrames; ++frame)
            {
                CommandList list;
                CG_REQUIRE(harness.NewList(QueueKind::Graphics, list));
                const auto start = std::chrono::steady_clock::now();
                CG_REQUIRE(CullInstancesReference(world.scene, view, nullptr, culled).IsOk());
                CG_REQUIRE(BeginPointPass(target, list));
                CG_REQUIRE(
                    list.SetVertexBuffer(kGpuDrivenInstanceStreamBinding, pipeline.VisibleBuffer())
                        .IsOk());
                const Span<const GpuDrawBatch> batches = world.scene.Batches();
                for (usize batch = 0; batch < batches.Size(); ++batch)
                {
                    for (u32 slot = 0; slot < culled.batchCounts[batch]; ++slot)
                    {
                        DrawIndexedArguments draw{};
                        draw.indexCount = batches[batch].indexCount;
                        draw.instanceCount = 1;
                        draw.firstIndex = batches[batch].firstIndex;
                        draw.firstInstance = batches[batch].visibleBase + slot;
                        CG_REQUIRE(list.DrawIndexed(draw).IsOk());
                    }
                }
                CG_REQUIRE(EndPointPass(target, list, false));
                CG_REQUIRE(list.End().IsOk());
                control[frame] = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
                // Recorded, never submitted: the control measures CPU work,
                // and its draws read an instance stream nothing has bound.
                list = CommandList{};
            }
            cpuDriven[sizeIndex] = Fastest(control, kControlFrames);
        }
        CG_REQUIRE(harness.device.WaitIdle().IsOk());
    }
    std::printf("adapter \"%s\": CPU ms per frame (fastest of %u)\n", adapter.name, kFrames);
    for (u32 sizeIndex = 0; sizeIndex < kSizeCount; ++sizeIndex)
    {
        if (kSizes[sizeIndex] <= 100000)
        {
            std::printf("  %8u instances: GPU-driven %.3f, CPU-driven control %.3f\n",
                        kSizes[sizeIndex], gpuDriven[sizeIndex], cpuDriven[sizeIndex]);
        }
        else
        {
            std::printf("  %8u instances: GPU-driven %.3f\n", kSizes[sizeIndex],
                        gpuDriven[sizeIndex]);
        }
    }
    // Flat: a thousand-fold increase in instances costs the CPU at most twice
    // as much, with a 0.25 ms floor for timer and scheduler noise at
    // sub-millisecond scales.
    CG_EXPECT_TRUE(gpuDriven[kSizeCount - 1] <= 2.0 * gpuDriven[0] + 0.25);
    // The control is what growing looks like: a hundredfold more instances
    // cost it at least twenty times more.
    CG_EXPECT_TRUE(cpuDriven[2] >= 20.0 * cpuDriven[0]);
}

// ---------------------------------------------------------------------------
// Light culling beside instance culling on the compute queue
// ---------------------------------------------------------------------------

// Both culling passes in one compute-queue submission, the draw on the
// graphics queue behind one fence: the build guide's async compute queue "for
// culling and light culling". The instance results equal the graphics-queue
// frame's and every cluster list contains what `BuildClusterLightLists` puts
// in it.
CG_TEST(gpu_driven_d3d12_light_and_instance_culling_share_the_compute_queue)
{
    Adapter adapters[8]{};
    const u32 count = ListAdapters(adapters);
    if (count == 0)
    {
        CG_SKIP("no Direct3D 12 adapter");
    }
    HeapAllocator heap;
    DebugLayerWatch watch{};
    cyber::core::CallbackSink sink{&WatchLog, &watch};
    CG_REQUIRE(cyber::core::DefaultLogger().AddSink(&sink));
    {
        const GpuCullView view = DefaultView();

        // A thousand point lights in front of the camera, as Phase 27's
        // clustered-lighting sample places them.
        GpuLightCullingDesc lightDesc{};
        lightDesc.grid.widthPixels = kViewportWidth;
        lightDesc.grid.heightPixels = kViewportHeight;
        lightDesc.grid.tileSizePixels = 32;
        lightDesc.grid.depthSlices = 24;
        lightDesc.grid.verticalFovRadians = kFov;
        lightDesc.grid.nearPlane = kNear;
        lightDesc.grid.farPlane = 200.0f;
        lightDesc.maxLights = 1000;
        lightDesc.maxLightsPerCluster = 256;
        DynamicArray<ClusterLightBound> lights(heap);
        CG_REQUIRE(lights.TryResize(lightDesc.maxLights));
        Random random(67);
        for (usize index = 0; index < lights.Size(); ++index)
        {
            lights[index].x = random.Range(-60.0f, 60.0f);
            lights[index].y = random.Range(-30.0f, 30.0f);
            lights[index].depth = random.Range(0.5f, 190.0f);
            lights[index].range = random.Range(0.5f, 6.0f);
        }
        ClusterLightLists cpuLists(heap);
        CG_REQUIRE(
            BuildClusterLightLists(Span<const ClusterLightBound>(lights.Data(), lights.Size()),
                                   lightDesc.grid, cpuLists)
                .IsOk());

        for (u32 adapterIndex = 0; adapterIndex < count; ++adapterIndex)
        {
            World world(heap);
            CG_REQUIRE(BuildWorld(world, InstancesFor(adapters[adapterIndex], 250000), 61));
            GpuCullOutput reference(heap);
            CG_REQUIRE(CullInstancesReference(world.scene, view, nullptr, reference).IsOk());
            Harness harness(heap);
            CG_REQUIRE(harness.Open(adapters[adapterIndex]));
            if (!harness.device.Queue(QueueKind::Compute).IsValid())
            {
                continue;
            }
            auto createdCulling =
                GpuCullingPipeline::Create(harness.device, PipelineDescFor(world.scene), heap);
            auto createdLights = GpuLightCullingPass::Create(harness.device, lightDesc, heap);
            CG_REQUIRE(static_cast<bool>(createdCulling) && static_cast<bool>(createdLights));
            GpuCullingPipeline culling = std::move(createdCulling.Value());
            GpuLightCullingPass lightCulling = std::move(createdLights.Value());
            PointTarget target;
            CG_REQUIRE(BuildPointTarget(harness.device, target));

            CommandList upload;
            CG_REQUIRE(harness.NewList(QueueKind::Graphics, upload));
            CG_REQUIRE(culling.RecordSceneUpload(upload, world.scene).IsOk());
            CG_REQUIRE(lightCulling
                           .RecordLightUpload(
                               upload, Span<const ClusterLightBound>(lights.Data(), lights.Size()))
                           .IsOk());
            CG_REQUIRE(harness.Run(upload));

            // Compute queue: instance culling, compaction, light culling.
            CommandList compute;
            CG_REQUIRE(harness.NewList(QueueKind::Compute, compute));
            CG_REQUIRE(culling.RecordCull(compute, view).IsOk());
            CG_REQUIRE(lightCulling.RecordCull(compute).IsOk());
            CG_EXPECT_EQ(compute.Stats().dispatches, u32{3});
            CG_REQUIRE(compute.End().IsOk());
            // Graphics queue: the draw, then the readbacks.
            CommandList graphics;
            CG_REQUIRE(harness.NewList(QueueKind::Graphics, graphics));
            CG_REQUIRE(culling.RecordAcquireForDraw(graphics).IsOk());
            CG_REQUIRE(BeginPointPass(target, graphics));
            CG_REQUIRE(culling.RecordDraw(graphics).IsOk());
            CG_REQUIRE(EndPointPass(target, graphics, true));
            CG_REQUIRE(culling.RecordReadback(graphics).IsOk());
            CG_REQUIRE(lightCulling.RecordReadback(graphics).IsOk());
            CG_REQUIRE(graphics.End().IsOk());

            const u64 culled = ++harness.fenceValue;
            const CommandListHandle computeHandle = compute.Handle();
            const FenceSignal computeSignal{harness.fence, culled};
            SubmitDesc computeSubmit{};
            computeSubmit.queue = QueueKind::Compute;
            computeSubmit.commandLists = Span<const CommandListHandle>(&computeHandle, 1);
            computeSubmit.signals = Span<const FenceSignal>(&computeSignal, 1);
            CG_REQUIRE(harness.device.Submit(computeSubmit).IsOk());
            const u64 drawn = ++harness.fenceValue;
            const CommandListHandle graphicsHandle = graphics.Handle();
            const FenceSignal graphicsSignal{harness.fence, drawn};
            const GpuWait wait{harness.fence, culled};
            SubmitDesc graphicsSubmit{};
            graphicsSubmit.queue = QueueKind::Graphics;
            graphicsSubmit.commandLists = Span<const CommandListHandle>(&graphicsHandle, 1);
            graphicsSubmit.signals = Span<const FenceSignal>(&graphicsSignal, 1);
            graphicsSubmit.gpuWaits = Span<const GpuWait>(&wait, 1);
            CG_REQUIRE(harness.device.Submit(graphicsSubmit).IsOk());
            CG_REQUIRE(harness.device.WaitFence(harness.fence, drawn).IsOk());

            GpuCullOutput gpu(heap);
            CG_REQUIRE(culling.ReadBack(gpu).IsOk());
            CG_EXPECT_EQ(gpu.counters.visible, reference.counters.visible);
            CG_EXPECT_TRUE(SameVisibleSets(world.scene, gpu, reference));
            DynamicArray<u8> lit(heap);
            CG_REQUIRE(ReadLitIds(harness.device, target, heap, lit));
            u32 drawnInstances = 0;
            for (usize id = 0; id < lit.Size(); ++id)
            {
                drawnInstances += lit[id];
            }
            CG_EXPECT_EQ(drawnInstances, gpu.counters.visible);

            DynamicArray<GpuClusterHeader> headers(heap);
            DynamicArray<u32> indices(heap);
            CG_REQUIRE(lightCulling.ReadBack(headers, indices).IsOk());
            CG_REQUIRE(headers.Size() == cpuLists.ClusterCount());
            u32 omitted = 0;
            u32 overflowing = 0;
            u64 gpuEntries = 0;
            u64 cpuEntries = 0;
            const u32 slots = lightCulling.MaxLightsPerCluster();
            for (u32 cluster = 0; cluster < cpuLists.ClusterCount(); ++cluster)
            {
                const GpuClusterHeader& header = headers[cluster];
                overflowing += header.overflow != 0 ? 1U : 0U;
                gpuEntries += header.count;
                const auto expected = cpuLists.Lights(cluster);
                cpuEntries += expected.Size();
                for (usize entry = 0; entry < expected.Size(); ++entry)
                {
                    bool found = false;
                    for (u32 slot = 0; slot < header.count && !found; ++slot)
                    {
                        found =
                            indices[static_cast<usize>(cluster) * slots + slot] == expected[entry];
                    }
                    omitted += found ? 0U : 1U;
                }
            }
            std::printf("adapter \"%s\": one compute submission culled %u instances and built "
                        "%u cluster lists (%llu GPU entries over %llu CPU entries)\n",
                        adapters[adapterIndex].name, gpu.counters.visible, cpuLists.ClusterCount(),
                        static_cast<unsigned long long>(gpuEntries),
                        static_cast<unsigned long long>(cpuEntries));
            CG_EXPECT_EQ(omitted, u32{0});
            CG_EXPECT_EQ(overflowing, u32{0});
            CG_REQUIRE(harness.device.WaitIdle().IsOk());
        }
    }
    (void)cyber::core::DefaultLogger().RemoveSink(&sink);
    if (watch.records != 0)
    {
        CG_FAIL(watch.last);
    }
}
