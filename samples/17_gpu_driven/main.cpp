// Phase 32 sample: a million instances culled on the GPU and drawn with one
// indirect call.
//
// A 1000 x 1000 voxel landscape — one cube per column — is described once to
// the GPU-driven renderer. Each frame the CPU writes the camera and records a
// culling dispatch, a compaction dispatch, and a single
// `DrawIndexedIndirectCount`; the GPU decides which of the million cubes are
// in view, which of three LODs each one draws, and how many indirect draws
// there are. The sample writes the frame as `gpu_driven.ppm` and prints what
// each side did.

#include <cyber/core/error.h>
#include <cyber/core/types.h>
#include <cyber/math/aabb.h>
#include <cyber/math/constants.h>
#include <cyber/math/mat4.h>
#include <cyber/math/vec3.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/renderer/gpu_culling_pipeline.h>
#include <cyber/renderer/gpu_driven.h>
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>

namespace
{

using namespace cyber::rhi;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;
using cyber::core::usize;
using cyber::math::Aabb;
using cyber::math::Mat4;
using cyber::math::Vec3;
using cyber::memory::Byte;
using cyber::memory::DynamicArray;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using cyber::renderer::GpuCullingPipeline;
using cyber::renderer::GpuCullingPipelineDesc;
using cyber::renderer::GpuCullOutput;
using cyber::renderer::GpuCullView;
using cyber::renderer::GpuDrawGroupDesc;
using cyber::renderer::GpuDrawLod;
using cyber::renderer::GpuDrivenScene;

constexpr u32 kWidth = 1280;
constexpr u32 kHeight = 720;
constexpr u32 kRowPitch = kWidth * 4;
constexpr u32 kSide = 1000;
constexpr u32 kInstances = kSide * kSide;
constexpr f32 kFov = cyber::math::kPi / 3.0f;
constexpr f32 kNear = 0.1f;
constexpr f32 kFar = 1600.0f;
constexpr u32 kTimedFrames = 16;

constexpr char kVertexShader[] = R"HLSL(
cbuffer Frame : register(b0, space0)
{
    float4 ViewProjection[4];
    float4 SunDirection;
    float4 CameraPosition;
};

StructuredBuffer<float4> Instances : register(t0, space0);

struct Input
{
    float3 position : TEXCOORD0;
    float3 normal : TEXCOORD1;
    uint instanceId : TEXCOORD2;
};

struct Output
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float3 color : COLOR;
    float3 world : WORLDPOS;
};

Output main(Input input)
{
    float4 placement = Instances[input.instanceId * 2];
    float4 tint = Instances[input.instanceId * 2 + 1];
    float3 world = placement.xyz + input.position * placement.w;
    Output output;
    output.position = ViewProjection[0] * world.x + ViewProjection[1] * world.y +
                      ViewProjection[2] * world.z + ViewProjection[3];
    output.normal = input.normal;
    output.color = tint.rgb;
    output.world = world;
    return output;
}
)HLSL";

constexpr char kPixelShader[] = R"HLSL(
cbuffer Frame : register(b0, space0)
{
    float4 ViewProjection[4];
    float4 SunDirection;
    float4 CameraPosition;
};

struct Input
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float3 color : COLOR;
    float3 world : WORLDPOS;
};

float4 main(Input input) : SV_Target
{
    float3 normal = normalize(input.normal);
    float diffuse = saturate(dot(normal, -SunDirection.xyz));
    float sky = 0.55 + 0.45 * normal.y;
    float3 lit = input.color * (diffuse * float3(1.05, 0.97, 0.86) + sky * float3(0.22, 0.28, 0.38));
    float distance = length(input.world - CameraPosition.xyz);
    float fog = 1.0 - exp(-distance * 0.0014);
    float3 color = lerp(lit, float3(0.60, 0.70, 0.84), fog);
    return float4(pow(saturate(color), 1.0 / 2.2), 1.0);
}
)HLSL";

struct Vertex
{
    f32 position[3];
    f32 normal[3];
};

struct FrameConstants
{
    f32 viewProjection[4][4];
    f32 sunDirection[4];
    f32 cameraPosition[4];
};

void Report(const char* what, const cyber::core::Error& error) noexcept
{
    std::fprintf(stderr, "%s failed: %s\n", what,
                 error.message != nullptr ? error.message : "no diagnostic");
}

[[nodiscard]] Span<const Byte> Bytes(const void* data, usize size) noexcept
{
    return Span<const Byte>(static_cast<const Byte*>(data), size);
}

// Flat-shaded unit meshes, one vertex per face corner so each face carries
// its own normal: a cube (LOD 0), an octahedron (LOD 1), a tetrahedron
// (LOD 2). All fit in the unit cube the instance's half-size scales.
void AddTriangle(DynamicArray<Vertex>& vertices, DynamicArray<u32>& indices, Vec3 a, Vec3 b,
                 Vec3 c) noexcept
{
    const Vec3 normal = cyber::math::Normalize(cyber::math::Cross(b - a, c - a));
    for (const Vec3& p : {a, b, c})
    {
        (void)indices.TryPushBack(static_cast<u32>(vertices.Size()));
        (void)vertices.TryPushBack(Vertex{{p.x, p.y, p.z}, {normal.x, normal.y, normal.z}});
    }
}

[[nodiscard]] GpuDrawLod BuildCube(DynamicArray<Vertex>& vertices,
                                   DynamicArray<u32>& indices) noexcept
{
    GpuDrawLod lod{0, static_cast<u32>(indices.Size()), 0};
    const u32 firstIndex = static_cast<u32>(indices.Size());
    const Vec3 c[8] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                       {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
    const u32 faces[6][4] = {{4, 5, 6, 7}, {1, 0, 3, 2}, {5, 1, 2, 6},
                             {0, 4, 7, 3}, {7, 6, 2, 3}, {0, 1, 5, 4}};
    for (const auto& face : faces)
    {
        AddTriangle(vertices, indices, c[face[0]], c[face[1]], c[face[2]]);
        AddTriangle(vertices, indices, c[face[0]], c[face[2]], c[face[3]]);
    }
    lod.indexCount = static_cast<u32>(indices.Size()) - firstIndex;
    return lod;
}

[[nodiscard]] GpuDrawLod BuildOctahedron(DynamicArray<Vertex>& vertices,
                                         DynamicArray<u32>& indices) noexcept
{
    GpuDrawLod lod{0, static_cast<u32>(indices.Size()), 0};
    const Vec3 axes[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    const u32 faces[8][3] = {{0, 2, 4}, {4, 2, 1}, {1, 2, 5}, {5, 2, 0},
                             {4, 3, 0}, {1, 3, 4}, {5, 3, 1}, {0, 3, 5}};
    for (const auto& face : faces)
    {
        AddTriangle(vertices, indices, axes[face[0]], axes[face[1]], axes[face[2]]);
    }
    lod.indexCount = 24;
    return lod;
}

[[nodiscard]] GpuDrawLod BuildTetrahedron(DynamicArray<Vertex>& vertices,
                                          DynamicArray<u32>& indices) noexcept
{
    GpuDrawLod lod{0, static_cast<u32>(indices.Size()), 0};
    const Vec3 p[4] = {{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}};
    AddTriangle(vertices, indices, p[0], p[2], p[1]);
    AddTriangle(vertices, indices, p[0], p[1], p[3]);
    AddTriangle(vertices, indices, p[0], p[3], p[2]);
    AddTriangle(vertices, indices, p[1], p[2], p[3]);
    lod.indexCount = 12;
    return lod;
}

// Terrain height in whole cubes, so neighbouring columns stack into steps.
[[nodiscard]] f32 Height(f32 x, f32 z) noexcept
{
    const f32 broad = 22.0f * std::sin(x * 0.0071f) * std::cos(z * 0.0063f);
    const f32 hills = 9.0f * std::sin(x * 0.031f + z * 0.017f) * std::cos(z * 0.027f - x * 0.011f);
    const f32 detail = 2.5f * std::sin(x * 0.13f) * std::sin(z * 0.11f);
    return std::floor(broad + hills + detail);
}

[[nodiscard]] Vec3 TerrainColor(f32 height, u32 hash) noexcept
{
    const f32 jitter = static_cast<f32>(hash & 0xFFU) / 255.0f * 0.08f - 0.04f;
    Vec3 color{};
    if (height < -14.0f)
    {
        color = Vec3{0.16f, 0.34f, 0.52f};
    }
    else if (height < -10.0f)
    {
        color = Vec3{0.76f, 0.70f, 0.50f};
    }
    else if (height < 8.0f)
    {
        color = Vec3{0.27f, 0.52f, 0.20f};
    }
    else if (height < 18.0f)
    {
        color = Vec3{0.46f, 0.42f, 0.38f};
    }
    else
    {
        color = Vec3{0.92f, 0.93f, 0.95f};
    }
    return color + Vec3{jitter};
}

[[nodiscard]] u32 Hash(u32 value) noexcept
{
    value ^= value >> 16U;
    value *= 0x7FEB352DU;
    value ^= value >> 15U;
    value *= 0x846CA68BU;
    value ^= value >> 16U;
    return value;
}

[[nodiscard]] bool OpenDevice(Device& device, HeapAllocator& heap,
                              char (&name)[kAdapterNameBytes]) noexcept
{
    const Backend* backend = D3D12Backend();
    if (backend == nullptr)
    {
        std::fprintf(stderr, "This build has no Direct3D 12 backend.\n");
        return false;
    }
    AdapterInfo adapters[8]{};
    const u32 count = std::min<u32>(backend->enumerateAdapters(adapters, 8), 8);
    // Hardware first; WARP only when it is all there is.
    u32 chosen = count;
    for (u32 index = 0; index < count && chosen == count; ++index)
    {
        chosen = adapters[index].vendorId != 0x1414 ? index : chosen;
    }
    chosen = chosen == count ? 0 : chosen;
    if (count == 0)
    {
        std::fprintf(stderr, "No Direct3D 12 adapter is available.\n");
        return false;
    }
    DeviceDesc desc{};
    desc.backend = backend;
    desc.adapterIndex = chosen;
    auto created = Device::Create(desc, heap);
    if (!created)
    {
        Report("Device::Create", created.Error());
        return false;
    }
    device = std::move(created.Value());
    std::snprintf(name, sizeof(name), "%s", adapters[chosen].name);
    return true;
}

[[nodiscard]] bool WritePpm(const char* path, const DynamicArray<u8>& rgba) noexcept
{
    std::FILE* file = nullptr;
#if defined(_MSC_VER)
    if (fopen_s(&file, path, "wb") != 0)
    {
        file = nullptr;
    }
#else
    file = std::fopen(path, "wb");
#endif
    if (file == nullptr)
    {
        return false;
    }
    std::fprintf(file, "P6\n%u %u\n255\n", kWidth, kHeight);
    bool ok = true;
    for (usize pixel = 0; pixel < static_cast<usize>(kWidth) * kHeight && ok; ++pixel)
    {
        ok = std::fwrite(&rgba[pixel * 4], 1, 3, file) == 3;
    }
    return std::fclose(file) == 0 && ok;
}

}  // namespace

int main()
{
    HeapAllocator heap;
    Device device;
    char adapterName[kAdapterNameBytes]{};
    if (!OpenDevice(device, heap, adapterName))
    {
        return 1;
    }

    // --- The scene, described once -------------------------------------
    DynamicArray<Vertex> vertices(heap);
    DynamicArray<u32> indices(heap);
    GpuDrawLod lods[3] = {BuildCube(vertices, indices), BuildOctahedron(vertices, indices),
                          BuildTetrahedron(vertices, indices)};
    GpuDrivenScene scene(heap);
    GpuDrawGroupDesc group{};
    group.lods = Span<const GpuDrawLod>(lods, 3);
    group.lodPixelThresholds[0] = 1.5f;
    group.lodPixelThresholds[1] = 0.6f;
    DynamicArray<f32> instanceData(heap);
    if (!scene.AddDrawGroup(group) || !scene.Reserve(kInstances) ||
        !instanceData.TryResize(static_cast<usize>(kInstances) * 8))
    {
        std::fprintf(stderr, "Could not allocate the scene.\n");
        return 1;
    }
    for (u32 row = 0; row < kSide; ++row)
    {
        for (u32 column = 0; column < kSide; ++column)
        {
            const u32 id = row * kSide + column;
            const f32 x = static_cast<f32>(column) - kSide * 0.5f;
            const f32 z = static_cast<f32>(row) - kSide * 0.5f;
            const f32 height = std::max(Height(x, z), -15.0f);
            const Vec3 center{x, height, z};
            const Vec3 color = TerrainColor(height, Hash(id));
            f32* data = &instanceData[static_cast<usize>(id) * 8];
            data[0] = center.x;
            data[1] = center.y;
            data[2] = center.z;
            data[3] = 0.5f;
            data[4] = color.x;
            data[5] = color.y;
            data[6] = color.z;
            data[7] = 1.0f;
            if (!scene.AddInstance(0, Aabb::FromCenterExtents(center, Vec3{0.5f}), id))
            {
                std::fprintf(stderr, "Could not add instance %u.\n", id);
                return 1;
            }
        }
    }
    if (!scene.Finalize())
    {
        std::fprintf(stderr, "Could not finalize the scene.\n");
        return 1;
    }

    GpuCullingPipelineDesc pipelineDesc{};
    pipelineDesc.maxInstances = kInstances;
    pipelineDesc.maxGroups = 1;
    pipelineDesc.maxBatches = static_cast<u32>(scene.Batches().Size());
    pipelineDesc.maxVisible = scene.VisibleCapacity();
    auto createdPipeline = GpuCullingPipeline::Create(device, pipelineDesc, heap);
    if (!createdPipeline)
    {
        Report("GpuCullingPipeline::Create", createdPipeline.Error());
        return 1;
    }
    GpuCullingPipeline culling = std::move(createdPipeline.Value());

    // --- Draw resources -------------------------------------------------
    const u64 vertexBytes = vertices.Size() * sizeof(Vertex);
    const u64 indexBytes = indices.Size() * sizeof(u32);
    const u64 instanceBytes = instanceData.Size() * sizeof(f32);
    const u64 constantOffset = vertexBytes + indexBytes + instanceBytes;
    const u64 alignedConstants = (constantOffset + 255) & ~u64{255};
    BufferDesc stagingDesc{};
    stagingDesc.usage = BufferUsage::HostVisible | BufferUsage::CopySource;
    stagingDesc.sizeBytes = alignedConstants + 256;
    BufferDesc vertexDesc{};
    vertexDesc.usage = BufferUsage::Vertex | BufferUsage::CopyDestination;
    vertexDesc.sizeBytes = vertexBytes;
    BufferDesc indexDesc{};
    indexDesc.usage = BufferUsage::Index | BufferUsage::CopyDestination;
    indexDesc.sizeBytes = indexBytes;
    BufferDesc instanceDesc{};
    instanceDesc.usage = BufferUsage::Storage | BufferUsage::CopyDestination;
    instanceDesc.sizeBytes = instanceBytes;
    instanceDesc.elementStride = sizeof(f32) * 4;
    BufferDesc constantDesc{};
    constantDesc.usage = BufferUsage::Uniform | BufferUsage::CopyDestination;
    constantDesc.sizeBytes = 256;
    BufferDesc readbackDesc{};
    readbackDesc.usage = BufferUsage::Readback | BufferUsage::CopyDestination;
    readbackDesc.sizeBytes = static_cast<u64>(kRowPitch) * kHeight;
    auto staging = device.CreateBuffer(stagingDesc);
    auto vertexBuffer = device.CreateBuffer(vertexDesc);
    auto indexBuffer = device.CreateBuffer(indexDesc);
    auto instanceBuffer = device.CreateBuffer(instanceDesc);
    auto constantBuffer = device.CreateBuffer(constantDesc);
    auto readback = device.CreateBuffer(readbackDesc);
    TextureDesc colorDesc{};
    colorDesc.format = Format::RGBA8Unorm;
    colorDesc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
    colorDesc.width = kWidth;
    colorDesc.height = kHeight;
    TextureDesc depthDesc{};
    depthDesc.format = Format::D32Float;
    depthDesc.usage = TextureUsage::DepthStencil;
    depthDesc.width = kWidth;
    depthDesc.height = kHeight;
    auto color = device.CreateTexture(colorDesc);
    auto depth = device.CreateTexture(depthDesc);
    if (!staging || !vertexBuffer || !indexBuffer || !instanceBuffer || !constantBuffer ||
        !readback || !color || !depth)
    {
        std::fprintf(stderr, "Could not create the draw resources.\n");
        return 1;
    }
    TextureViewDesc colorViewDesc{};
    colorViewDesc.texture = color.Value();
    colorViewDesc.format = Format::RGBA8Unorm;
    TextureViewDesc depthViewDesc{};
    depthViewDesc.texture = depth.Value();
    depthViewDesc.format = Format::D32Float;
    depthViewDesc.range.aspect = TextureAspect::Depth;
    auto colorView = device.CreateTextureView(colorViewDesc);
    auto depthView = device.CreateTextureView(depthViewDesc);
    if (!colorView || !depthView)
    {
        std::fprintf(stderr, "Could not create the attachment views.\n");
        return 1;
    }

    BindGroupLayoutEntry entries[2]{};
    entries[0].binding = 0;
    entries[0].type = BindingType::UniformBuffer;
    entries[0].visibility =
        ShaderStageBit(ShaderStage::Vertex) | ShaderStageBit(ShaderStage::Pixel);
    entries[1].binding = 1;
    entries[1].type = BindingType::ReadOnlyStorageBuffer;
    entries[1].visibility = ShaderStageBit(ShaderStage::Vertex);
    BindGroupLayoutDesc groupLayoutDesc{};
    groupLayoutDesc.entries = Span<const BindGroupLayoutEntry>(entries, 2);
    auto groupLayout = device.CreateBindGroupLayout(groupLayoutDesc);
    if (!groupLayout)
    {
        Report("CreateBindGroupLayout", groupLayout.Error());
        return 1;
    }
    BindingResource resources[2]{};
    resources[0].type = BindingType::UniformBuffer;
    resources[0].buffer = constantBuffer.Value();
    resources[1].type = BindingType::ReadOnlyStorageBuffer;
    resources[1].buffer = instanceBuffer.Value();
    BindGroupDesc groupDesc{};
    groupDesc.layout = groupLayout.Value();
    groupDesc.resources = Span<const BindingResource>(resources, 2);
    auto bindGroup = device.CreateBindGroup(groupDesc);
    PipelineLayoutDesc layoutDesc{};
    layoutDesc.groups[static_cast<u32>(BindGroupSlot::PerFrame)] = groupLayout.Value();
    auto layout = device.CreatePipelineLayout(layoutDesc);
    ShaderDesc vertexShaderDesc{};
    vertexShaderDesc.stage = ShaderStage::Vertex;
    vertexShaderDesc.code = Bytes(kVertexShader, std::strlen(kVertexShader));
    vertexShaderDesc.entryPoint = "main";
    ShaderDesc pixelShaderDesc{};
    pixelShaderDesc.stage = ShaderStage::Pixel;
    pixelShaderDesc.code = Bytes(kPixelShader, std::strlen(kPixelShader));
    pixelShaderDesc.entryPoint = "main";
    auto vertexShader = device.CreateShader(vertexShaderDesc);
    auto pixelShader = device.CreateShader(pixelShaderDesc);
    if (!bindGroup || !layout || !vertexShader || !pixelShader)
    {
        std::fprintf(stderr, "Could not create the shaders or bindings.\n");
        return 1;
    }
    VertexBinding bindings[2]{};
    bindings[0].strideBytes = sizeof(Vertex);
    bindings[0].rate = VertexInputRate::PerVertex;
    bindings[1].strideBytes = sizeof(u32);
    bindings[1].rate = VertexInputRate::PerInstance;
    VertexAttribute attributes[3]{};
    attributes[0] = VertexAttribute{0, 0, Format::R32G32B32Float, 0};
    attributes[1] = VertexAttribute{1, 0, Format::R32G32B32Float, 12};
    attributes[2] =
        VertexAttribute{2, cyber::renderer::kGpuDrivenInstanceStreamBinding, Format::R32Uint, 0};
    ColorAttachmentState attachment{};
    attachment.format = Format::RGBA8Unorm;
    GraphicsPipelineDesc graphicsDesc{};
    graphicsDesc.layout = layout.Value();
    graphicsDesc.vertexShader = vertexShader.Value();
    graphicsDesc.pixelShader = pixelShader.Value();
    graphicsDesc.vertexLayout.bindings = Span<const VertexBinding>(bindings, 2);
    graphicsDesc.vertexLayout.attributes = Span<const VertexAttribute>(attributes, 3);
    graphicsDesc.rasterizer.cullMode = CullMode::Back;
    graphicsDesc.depthStencil.depthCompare = CompareFunction::GreaterEqual;
    graphicsDesc.depthStencilFormat = Format::D32Float;
    graphicsDesc.colorAttachments = Span<const ColorAttachmentState>(&attachment, 1);
    auto graphics = device.CreateGraphicsPipeline(graphicsDesc);
    if (!graphics)
    {
        Report("CreateGraphicsPipeline", graphics.Error());
        return 1;
    }

    // --- The camera -------------------------------------------------------
    const Vec3 eye{-90.0f, 62.0f, 470.0f};
    const Vec3 target{60.0f, -6.0f, 250.0f};
    const Mat4 viewProjection =
        Mat4::PerspectiveReverseZ(kFov, static_cast<f32>(kWidth) / static_cast<f32>(kHeight), kNear,
                                  kFar) *
        Mat4::LookAt(eye, target, Vec3{0.0f, 1.0f, 0.0f});
    const GpuCullView view =
        cyber::renderer::MakeGpuCullView(viewProjection, eye, kFov, kHeight, kNear);
    FrameConstants constants{};
    for (u32 column = 0; column < 4; ++column)
    {
        constants.viewProjection[column][0] = viewProjection[column].x;
        constants.viewProjection[column][1] = viewProjection[column].y;
        constants.viewProjection[column][2] = viewProjection[column].z;
        constants.viewProjection[column][3] = viewProjection[column].w;
    }
    const Vec3 sun = cyber::math::Normalize(Vec3{0.45f, -0.75f, -0.35f});
    constants.sunDirection[0] = sun.x;
    constants.sunDirection[1] = sun.y;
    constants.sunDirection[2] = sun.z;
    constants.cameraPosition[0] = eye.x;
    constants.cameraPosition[1] = eye.y;
    constants.cameraPosition[2] = eye.z;

    if (!device.WriteBuffer(staging.Value(), 0, Bytes(vertices.Data(), vertexBytes)) ||
        !device.WriteBuffer(staging.Value(), vertexBytes, Bytes(indices.Data(), indexBytes)) ||
        !device.WriteBuffer(staging.Value(), vertexBytes + indexBytes,
                            Bytes(instanceData.Data(), instanceBytes)) ||
        !device.WriteBuffer(staging.Value(), alignedConstants,
                            Bytes(&constants, sizeof(constants))))
    {
        std::fprintf(stderr, "Could not write the staging buffer.\n");
        return 1;
    }
    auto fenceResult = device.CreateFence(FenceDesc{});
    if (!fenceResult)
    {
        Report("CreateFence", fenceResult.Error());
        return 1;
    }
    const FenceHandle fence = fenceResult.Value();
    u64 fenceValue = 0;

    // --- Frames -----------------------------------------------------------
    double recordMilliseconds[kTimedFrames]{};
    double frameMilliseconds[kTimedFrames]{};
    u32 drawCalls = 0;
    for (u32 frame = 0; frame <= kTimedFrames; ++frame)
    {
        auto created =
            device.CreateCommandList(CommandListDesc{QueueKind::Graphics, "gpu-driven frame"});
        if (!created || !created.Value().Begin())
        {
            std::fprintf(stderr, "Could not begin a command list.\n");
            return 1;
        }
        CommandList list = std::move(created.Value());
        const bool first = frame == 0;
        if (first)
        {
            const BufferBarrier upload[] = {
                {staging.Value(), ResourceState::Undefined, ResourceState::CopySource},
                {vertexBuffer.Value(), ResourceState::Undefined, ResourceState::CopyDestination},
                {indexBuffer.Value(), ResourceState::Undefined, ResourceState::CopyDestination},
                {instanceBuffer.Value(), ResourceState::Undefined, ResourceState::CopyDestination},
                {constantBuffer.Value(), ResourceState::Undefined, ResourceState::CopyDestination},
                {readback.Value(), ResourceState::Undefined, ResourceState::CopyDestination},
            };
            const BufferBarrier ready[] = {
                {vertexBuffer.Value(), ResourceState::CopyDestination, ResourceState::VertexBuffer},
                {indexBuffer.Value(), ResourceState::CopyDestination, ResourceState::IndexBuffer},
                {instanceBuffer.Value(), ResourceState::CopyDestination, ResourceState::ShaderRead},
                {constantBuffer.Value(), ResourceState::CopyDestination,
                 ResourceState::ConstantBuffer},
            };
            const TextureBarrier targets[] = {
                {color.Value(), SubresourceRange{}, ResourceState::Undefined,
                 ResourceState::RenderTarget},
                {depth.Value(), SubresourceRange{0, 0, 0, 0, TextureAspect::Depth},
                 ResourceState::Undefined, ResourceState::DepthWrite},
            };
            if (!list.Barrier(Span<const BufferBarrier>(upload, 6)) ||
                !list.CopyBuffer(
                    BufferCopy{staging.Value(), 0, vertexBuffer.Value(), 0, vertexBytes}) ||
                !list.CopyBuffer(
                    BufferCopy{staging.Value(), vertexBytes, indexBuffer.Value(), 0, indexBytes}) ||
                !list.CopyBuffer(BufferCopy{staging.Value(), vertexBytes + indexBytes,
                                            instanceBuffer.Value(), 0, instanceBytes}) ||
                !list.CopyBuffer(BufferCopy{staging.Value(), alignedConstants,
                                            constantBuffer.Value(), 0, sizeof(constants)}) ||
                !list.Barrier(Span<const BufferBarrier>(ready, 4),
                              Span<const TextureBarrier>(targets, 2)) ||
                !culling.RecordSceneUpload(list, scene))
            {
                std::fprintf(stderr, "Could not record the uploads.\n");
                return 1;
            }
        }
        else
        {
            const TextureBarrier toTarget{color.Value(), SubresourceRange{},
                                          ResourceState::CopySource, ResourceState::RenderTarget};
            if (!list.Barrier(Span<const BufferBarrier>{},
                              Span<const TextureBarrier>(&toTarget, 1)))
            {
                return 1;
            }
        }

        // What the CPU pays every frame, whatever the instance count.
        const auto recordStart = std::chrono::steady_clock::now();
        ColorAttachment colorAttachment{};
        colorAttachment.view = colorView.Value();
        colorAttachment.loadOp = LoadOp::Clear;
        colorAttachment.storeOp = StoreOp::Store;
        colorAttachment.clearValue = ClearColorValue::Of(0.60f, 0.70f, 0.84f, 1.0f);
        RenderPassDesc pass{};
        pass.colorAttachments = Span<const ColorAttachment>(&colorAttachment, 1);
        pass.hasDepthStencil = true;
        pass.depthStencil.view = depthView.Value();
        pass.depthStencil.depthLoadOp = LoadOp::Clear;
        pass.depthStencil.depthStoreOp = StoreOp::DontCare;
        pass.depthStencil.clearValue = ClearDepthStencilValue::Of(0.0f);
        pass.label = "GPU-driven terrain";
        const u32 drawsBefore = list.Stats().draws;
        if (auto culled = culling.RecordCull(list, view); !culled)
        {
            Report("RecordCull", culled.Error());
            return 1;
        }
        if (!culling.RecordAcquireForDraw(list) || !list.BeginRenderPass(pass) ||
            !list.SetPipeline(graphics.Value()) ||
            !list.SetBindGroup(BindGroupSlot::PerFrame, bindGroup.Value()) ||
            !list.SetViewport(Viewport::Of(static_cast<f32>(kWidth), static_cast<f32>(kHeight))) ||
            !list.SetScissor(ScissorRect{0, 0, kWidth, kHeight}) ||
            !list.SetVertexBuffer(0, vertexBuffer.Value()) ||
            !list.SetIndexBuffer(indexBuffer.Value(), 0, IndexFormat::Uint32))
        {
            std::fprintf(stderr, "Could not open the terrain pass.\n");
            return 1;
        }
        if (auto drawn = culling.RecordDraw(list); !drawn)
        {
            Report("RecordDraw", drawn.Error());
            return 1;
        }
        drawCalls = list.Stats().draws - drawsBefore;
        if (!list.EndRenderPass())
        {
            return 1;
        }
        const double recorded = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - recordStart)
                                    .count();

        const TextureBarrier toCopy{color.Value(), SubresourceRange{}, ResourceState::RenderTarget,
                                    ResourceState::CopySource};
        BufferTextureCopy copy{};
        copy.buffer = readback.Value();
        copy.bufferRowPitch = kRowPitch;
        copy.texture = color.Value();
        copy.width = kWidth;
        copy.height = kHeight;
        if (!list.Barrier(Span<const BufferBarrier>{}, Span<const TextureBarrier>(&toCopy, 1)) ||
            !list.CopyTextureToBuffer(copy) || !culling.RecordReadback(list) || !list.End())
        {
            std::fprintf(stderr, "Could not finish the frame.\n");
            return 1;
        }
        const CommandListHandle handle = list.Handle();
        const FenceSignal signal{fence, ++fenceValue};
        SubmitDesc submit{};
        submit.queue = QueueKind::Graphics;
        submit.commandLists = Span<const CommandListHandle>(&handle, 1);
        submit.signals = Span<const FenceSignal>(&signal, 1);
        const auto submitStart = std::chrono::steady_clock::now();
        if (!device.Submit(submit) || !device.WaitFence(fence, fenceValue))
        {
            std::fprintf(stderr, "The frame did not complete.\n");
            return 1;
        }
        const double gpu = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - submitStart)
                               .count();
        if (!first)
        {
            recordMilliseconds[frame - 1] = recorded;
            frameMilliseconds[frame - 1] = gpu;
        }
    }

    GpuCullOutput output(heap);
    DynamicArray<u8> pixels(heap);
    if (!culling.ReadBack(output) || !pixels.TryResize(static_cast<usize>(kRowPitch) * kHeight) ||
        !device.ReadBuffer(readback.Value(), 0,
                           Span<Byte>(reinterpret_cast<Byte*>(pixels.Data()), pixels.Size())))
    {
        std::fprintf(stderr, "Could not read the results back.\n");
        return 1;
    }
    std::sort(recordMilliseconds, recordMilliseconds + kTimedFrames);
    std::sort(frameMilliseconds, frameMilliseconds + kTimedFrames);
    u32 lodCounts[3]{};
    for (usize batch = 0; batch < output.batchCounts.Size() && batch < 3; ++batch)
    {
        lodCounts[batch] = output.batchCounts[batch];
    }
    std::printf("adapter:   %s\n", adapterName);
    std::printf("instances: %u described, %u in view (%u frustum-culled)\n", kInstances,
                output.counters.visible, output.counters.frustumCulled);
    std::printf("LODs:      %u cubes, %u octahedra, %u tetrahedra\n", lodCounts[0], lodCounts[1],
                lodCounts[2]);
    std::printf("commands:  %u culling dispatches, %u draw call carrying %u indirect draws\n",
                culling.Stats().cullDispatches, drawCalls, output.counters.drawCount);
    std::printf("CPU:       %.3f ms median to record culling and the draw\n",
                recordMilliseconds[kTimedFrames / 2]);
    std::printf("frame:     %.2f ms median submit to completion\n",
                frameMilliseconds[kTimedFrames / 2]);
    if (!WritePpm("gpu_driven.ppm", pixels))
    {
        std::fprintf(stderr, "Could not write gpu_driven.ppm.\n");
        return 1;
    }
    std::printf("wrote gpu_driven.ppm\n");

    (void)device.WaitIdle();
    (void)device.DestroyGraphicsPipeline(graphics.Value());
    (void)device.DestroyShader(vertexShader.Value());
    (void)device.DestroyShader(pixelShader.Value());
    (void)device.DestroyPipelineLayout(layout.Value());
    (void)device.DestroyBindGroup(bindGroup.Value());
    (void)device.DestroyBindGroupLayout(groupLayout.Value());
    (void)device.DestroyTextureView(colorView.Value());
    (void)device.DestroyTextureView(depthView.Value());
    (void)device.DestroyTexture(color.Value());
    (void)device.DestroyTexture(depth.Value());
    const BufferHandle buffers[] = {staging.Value(),        vertexBuffer.Value(),
                                    indexBuffer.Value(),    instanceBuffer.Value(),
                                    constantBuffer.Value(), readback.Value()};
    for (const BufferHandle buffer : buffers)
    {
        (void)device.DestroyBuffer(buffer);
    }
    (void)device.DestroyFence(fence);
    return 0;
}
