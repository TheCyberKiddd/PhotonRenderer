#include <cyber/renderer/gpu_light_culling.h>
#include <cyber/rhi/binding.h>
#include <cyber/rhi/buffer.h>
#include <cyber/rhi/capabilities.h>
#include <cyber/rhi/pipeline.h>
#include <cyber/rhi/resource_state.h>
#include <cyber/rhi/shader.h>

#include <cmath>
#include <cstring>
#include <new>

namespace cyber::renderer
{

namespace
{

using cyber::core::ErrorCode;
using cyber::core::f32;
using cyber::core::u64;
using cyber::core::usize;
using cyber::memory::Byte;
using cyber::memory::Span;
using cyber::rhi::BufferBarrier;
using cyber::rhi::BufferHandle;
using cyber::rhi::BufferUsage;
using cyber::rhi::ResourceState;
using cyber::rhi::ShaderStage;

// The cluster light-list build, one thread per cluster. The projection is
// `BuildClusterLightLists`'s; the slice range is one slice wider each way so
// that the single-precision logarithm here cannot drop a light the CPU's
// double-precision one keeps, and the tile range is widened by a pixel as the
// CPU's is.
constexpr char kLightCullingShader[] = R"HLSL(
cbuffer LightCullParams : register(b0, space0)
{
    uint4 Grid0;       // x width, y height, z tile size, w depth slices
    uint4 Grid1;       // x tiles X, y tiles Y, z lights, w slots per cluster
    float4 Projection; // x tan(fovY / 2), y near, z far, w aspect
};

// x, y view-space position; z positive view depth; w range.
StructuredBuffer<float4> Lights : register(t0, space0);
RWStructuredBuffer<uint2> Headers : register(u0, space0);
RWStructuredBuffer<uint> Indices : register(u1, space0);

uint DepthSlice(float depth)
{
    float ratio = log(depth / Projection.y) / log(Projection.z / Projection.y);
    return min((uint)max(floor(ratio * Grid0.w), 0.0), Grid0.w - 1);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID)
{
    uint tilesX = Grid1.x;
    uint tilesY = Grid1.y;
    uint clusterCount = tilesX * tilesY * Grid0.w;
    uint cluster = dispatchId.x;
    if (cluster >= clusterCount)
        return;
    uint tileX = cluster % tilesX;
    uint tileY = (cluster / tilesX) % tilesY;
    uint slice = cluster / (tilesX * tilesY);
    uint count = 0;
    uint overflow = 0;

    for (uint lightIndex = 0; lightIndex < Grid1.z; ++lightIndex)
    {
        float4 sphere = Lights[lightIndex];
        float depthMin = sphere.z - sphere.w;
        float depthMax = sphere.z + sphere.w;
        if (depthMax < Projection.y || depthMin > Projection.z)
            continue;

        uint minSlice = DepthSlice(max(depthMin, Projection.y));
        uint maxSlice = DepthSlice(min(depthMax, Projection.z));
        minSlice = minSlice > 0 ? minSlice - 1 : 0;
        maxSlice = min(maxSlice + 1, Grid0.w - 1);
        if (slice < minSlice || slice > maxSlice)
            continue;

        float2 pixelMin = float2(0.0, 0.0);
        float2 pixelMax = float2(Grid0.x, Grid0.y);
        if (depthMin > Projection.y)
        {
            pixelMin = float2(3.402823466e+38, 3.402823466e+38);
            pixelMax = float2(-3.402823466e+38, -3.402823466e+38);
            for (uint depthCorner = 0; depthCorner < 2; ++depthCorner)
            {
                float depth = depthCorner == 0 ? depthMin : depthMax;
                for (uint axisCorner = 0; axisCorner < 2; ++axisCorner)
                {
                    float x = sphere.x + (axisCorner == 0 ? -sphere.w : sphere.w);
                    float pixelX =
                        (0.5 + 0.5 * x / (depth * Projection.x * Projection.w)) * Grid0.x;
                    pixelMin.x = min(pixelMin.x, pixelX);
                    pixelMax.x = max(pixelMax.x, pixelX);
                    float y = sphere.y + (axisCorner == 0 ? -sphere.w : sphere.w);
                    float pixelY = (0.5 - 0.5 * y / (depth * Projection.x)) * Grid0.y;
                    pixelMin.y = min(pixelMin.y, pixelY);
                    pixelMax.y = max(pixelMax.y, pixelY);
                }
            }
        }
        if (pixelMax.x < 0.0 || pixelMin.x > Grid0.x || pixelMax.y < 0.0 || pixelMin.y > Grid0.y)
            continue;

        uint minTileX = (uint)clamp(floor((pixelMin.x - 1.0) / Grid0.z), 0.0, (float)(tilesX - 1));
        uint maxTileX = (uint)clamp(floor((pixelMax.x + 1.0) / Grid0.z), 0.0, (float)(tilesX - 1));
        uint minTileY = (uint)clamp(floor((pixelMin.y - 1.0) / Grid0.z), 0.0, (float)(tilesY - 1));
        uint maxTileY = (uint)clamp(floor((pixelMax.y + 1.0) / Grid0.z), 0.0, (float)(tilesY - 1));
        if (tileX < minTileX || tileX > maxTileX || tileY < minTileY || tileY > maxTileY)
            continue;

        if (count < Grid1.w)
            Indices[cluster * Grid1.w + count] = lightIndex;
        else
            overflow = 1;
        ++count;
    }
    Headers[cluster] = uint2(min(count, Grid1.w), overflow);
}
)HLSL";

struct alignas(16) LightCullParams
{
    u32 grid0[4];
    u32 grid1[4];
    f32 projection[4];
};
static_assert(sizeof(LightCullParams) == 48, "LightCullParams mirrors the HLSL cbuffer");

inline constexpr u64 kParamBytes = 256;
inline constexpr u32 kThreadsPerGroup = 64;

[[nodiscard]] Status<Error> Fail(ErrorCode code, const char* message) noexcept
{
    return Status<Error>::Err(Error{code, message});
}

[[nodiscard]] Span<const Byte> AsBytes(const void* data, usize size) noexcept
{
    return Span<const Byte>(static_cast<const Byte*>(data), size);
}

}  // namespace

struct GpuLightCullingPass::Impl
{
    explicit Impl(cyber::rhi::Device& owner, cyber::memory::Allocator& owningAllocator) noexcept
        : device(&owner), allocator(&owningAllocator)
    {
    }

    cyber::rhi::Device* device = nullptr;
    cyber::memory::Allocator* allocator = nullptr;
    GpuLightCullingDesc desc{};
    u32 tilesX = 0;
    u32 tilesY = 0;
    u32 clusterCount = 0;
    u32 lightCount = 0;
    bool lightsUploaded = false;
    bool stagingReady = false;

    BufferHandle params{};
    BufferHandle lights{};
    BufferHandle headers{};
    BufferHandle indices{};
    BufferHandle staging{};
    BufferHandle headerReadback{};
    BufferHandle indexReadback{};
    ResourceState paramsState = ResourceState::Undefined;
    ResourceState lightsState = ResourceState::Undefined;
    ResourceState headersState = ResourceState::Undefined;
    ResourceState indicesState = ResourceState::Undefined;

    cyber::rhi::ShaderHandle shader{};
    cyber::rhi::BindGroupLayoutHandle groupLayout{};
    cyber::rhi::BindGroupHandle group{};
    cyber::rhi::PipelineLayoutHandle layout{};
    cyber::rhi::PipelineHandle pipeline{};

    [[nodiscard]] Status<Error> MakeBuffer(BufferHandle& out, BufferUsage usage, u64 sizeBytes,
                                           u32 stride, const char* label) noexcept
    {
        cyber::rhi::BufferDesc bufferDesc{};
        bufferDesc.usage = usage;
        bufferDesc.sizeBytes = sizeBytes < 16 ? 16 : sizeBytes;
        bufferDesc.elementStride = stride;
        bufferDesc.label = label;
        auto created = device->CreateBuffer(bufferDesc);
        if (!created)
        {
            return Status<Error>::Err(created.Error());
        }
        out = created.Value();
        return Status<Error>::Ok();
    }

    // Records one buffer transition and updates its tracked state.
    [[nodiscard]] Status<Error> Move(cyber::rhi::CommandList& list, BufferHandle buffer,
                                     ResourceState& state, ResourceState to) noexcept
    {
        if (state == to)
        {
            return Status<Error>::Ok();
        }
        const BufferBarrier barrier{buffer, state, to};
        const Status<Error> recorded = list.Barrier(Span<const BufferBarrier>(&barrier, 1));
        if (recorded)
        {
            state = to;
        }
        return recorded;
    }

    [[nodiscard]] Status<Error> PrepareStaging(cyber::rhi::CommandList& list) noexcept
    {
        if (stagingReady)
        {
            return Status<Error>::Ok();
        }
        const BufferBarrier barriers[] = {
            {staging, ResourceState::Undefined, ResourceState::CopySource},
            {headerReadback, ResourceState::Undefined, ResourceState::CopyDestination},
            {indexReadback, ResourceState::Undefined, ResourceState::CopyDestination},
        };
        const Status<Error> recorded = list.Barrier(Span<const BufferBarrier>(barriers, 3));
        stagingReady = recorded.IsOk();
        return recorded;
    }

    void DestroyAll() noexcept
    {
        cyber::rhi::Device& rhi = *device;
        if (group.IsValid())
        {
            (void)rhi.DestroyBindGroup(group);
        }
        if (pipeline.IsValid())
        {
            (void)rhi.DestroyComputePipeline(pipeline);
        }
        if (layout.IsValid())
        {
            (void)rhi.DestroyPipelineLayout(layout);
        }
        if (groupLayout.IsValid())
        {
            (void)rhi.DestroyBindGroupLayout(groupLayout);
        }
        if (shader.IsValid())
        {
            (void)rhi.DestroyShader(shader);
        }
        const BufferHandle buffers[] = {params,  lights,         headers,      indices,
                                        staging, headerReadback, indexReadback};
        for (const BufferHandle buffer : buffers)
        {
            if (buffer.IsValid())
            {
                (void)rhi.DestroyBuffer(buffer);
            }
        }
    }
};

GpuLightCullingPass::GpuLightCullingPass(Impl* impl) noexcept : m_impl(impl) {}

GpuLightCullingPass::~GpuLightCullingPass()
{
    Destroy();
}

GpuLightCullingPass::GpuLightCullingPass(GpuLightCullingPass&& other) noexcept
    : m_impl(other.m_impl)
{
    other.m_impl = nullptr;
}

GpuLightCullingPass& GpuLightCullingPass::operator=(GpuLightCullingPass&& other) noexcept
{
    if (this != &other)
    {
        Destroy();
        m_impl = other.m_impl;
        other.m_impl = nullptr;
    }
    return *this;
}

void GpuLightCullingPass::Destroy() noexcept
{
    if (m_impl == nullptr)
    {
        return;
    }
    m_impl->DestroyAll();
    cyber::memory::Allocator* allocator = m_impl->allocator;
    m_impl->~Impl();
    allocator->Deallocate(m_impl);
    m_impl = nullptr;
}

Result<GpuLightCullingPass, Error>
GpuLightCullingPass::Create(cyber::rhi::Device& device, const GpuLightCullingDesc& desc,
                            cyber::memory::Allocator& allocator) noexcept
{
    using ResultType = Result<GpuLightCullingPass, Error>;
    if (!device.Capabilities().features.Has(cyber::rhi::Feature::Compute))
    {
        return ResultType::Err(Error{ErrorCode::Unsupported, "gpu light culling: needs compute"});
    }
    const ClusterGridDesc& grid = desc.grid;
    if (grid.widthPixels == 0 || grid.heightPixels == 0 || grid.tileSizePixels == 0 ||
        grid.depthSlices == 0 || !(grid.nearPlane > 0.0f) || !(grid.farPlane > grid.nearPlane) ||
        !(grid.verticalFovRadians > 0.0f) || desc.maxLights == 0 || desc.maxLightsPerCluster == 0)
    {
        return ResultType::Err(
            Error{ErrorCode::InvalidArgument, "gpu light culling: invalid grid or capacity"});
    }
    void* memory = allocator.Allocate(sizeof(Impl), alignof(Impl));
    if (memory == nullptr)
    {
        return ResultType::Err(Error{ErrorCode::Internal, "gpu light culling: allocation failed"});
    }
    GpuLightCullingPass pass(new (memory) Impl(device, allocator));
    Impl& impl = *pass.m_impl;
    impl.desc = desc;
    impl.tilesX = (grid.widthPixels + grid.tileSizePixels - 1) / grid.tileSizePixels;
    impl.tilesY = (grid.heightPixels + grid.tileSizePixels - 1) / grid.tileSizePixels;
    const u64 clusters = static_cast<u64>(impl.tilesX) * impl.tilesY * grid.depthSlices;
    if (clusters * desc.maxLightsPerCluster > 0xFFFFFFFFULL)
    {
        return ResultType::Err(
            Error{ErrorCode::OutOfRange, "gpu light culling: the index list is too large"});
    }
    impl.clusterCount = static_cast<u32>(clusters);

    const u64 lightBytes = static_cast<u64>(desc.maxLights) * sizeof(f32) * 4;
    const u64 headerBytes = clusters * sizeof(GpuClusterHeader);
    const u64 indexBytes = clusters * desc.maxLightsPerCluster * sizeof(u32);
    const BufferUsage readback = BufferUsage::Readback | BufferUsage::CopyDestination;
    const Status<Error> buffers[] = {
        impl.MakeBuffer(impl.params, BufferUsage::Uniform | BufferUsage::CopyDestination,
                        kParamBytes, 0, "gpu light culling params"),
        impl.MakeBuffer(impl.lights, BufferUsage::Storage | BufferUsage::CopyDestination,
                        lightBytes, sizeof(f32) * 4, "gpu light culling lights"),
        impl.MakeBuffer(impl.headers, BufferUsage::Storage | BufferUsage::CopySource, headerBytes,
                        sizeof(GpuClusterHeader), "gpu light culling headers"),
        impl.MakeBuffer(impl.indices, BufferUsage::Storage | BufferUsage::CopySource, indexBytes,
                        sizeof(u32), "gpu light culling indices"),
        impl.MakeBuffer(impl.staging, BufferUsage::HostVisible | BufferUsage::CopySource,
                        kParamBytes + lightBytes, 0, "gpu light culling staging"),
        impl.MakeBuffer(impl.headerReadback, readback, headerBytes, 0,
                        "gpu light culling headers readback"),
        impl.MakeBuffer(impl.indexReadback, readback, indexBytes, 0,
                        "gpu light culling indices readback"),
    };
    for (const Status<Error>& status : buffers)
    {
        if (!status)
        {
            return ResultType::Err(status.Error());
        }
    }

    cyber::rhi::BindGroupLayoutEntry entries[4]{};
    const cyber::rhi::BindingType types[4] = {
        cyber::rhi::BindingType::UniformBuffer, cyber::rhi::BindingType::ReadOnlyStorageBuffer,
        cyber::rhi::BindingType::StorageBuffer, cyber::rhi::BindingType::StorageBuffer};
    const u32 registers[4] = {0, 0, 0, 1};
    const BufferHandle resourcesOf[4] = {impl.params, impl.lights, impl.headers, impl.indices};
    cyber::rhi::BindingResource resources[4]{};
    for (u32 index = 0; index < 4; ++index)
    {
        entries[index].binding = index;
        entries[index].type = types[index];
        entries[index].visibility = cyber::rhi::ShaderStageBit(ShaderStage::Compute);
        entries[index].shaderRegister = registers[index];
        resources[index].type = types[index];
        resources[index].buffer = resourcesOf[index];
    }
    cyber::rhi::BindGroupLayoutDesc groupLayoutDesc{};
    groupLayoutDesc.entries = Span<const cyber::rhi::BindGroupLayoutEntry>(entries, 4);
    auto groupLayout = device.CreateBindGroupLayout(groupLayoutDesc);
    if (!groupLayout)
    {
        return ResultType::Err(groupLayout.Error());
    }
    impl.groupLayout = groupLayout.Value();
    cyber::rhi::BindGroupDesc groupDesc{};
    groupDesc.layout = impl.groupLayout;
    groupDesc.resources = Span<const cyber::rhi::BindingResource>(resources, 4);
    auto group = device.CreateBindGroup(groupDesc);
    if (!group)
    {
        return ResultType::Err(group.Error());
    }
    impl.group = group.Value();

    cyber::rhi::ShaderDesc shaderDesc{};
    shaderDesc.stage = ShaderStage::Compute;
    shaderDesc.format = cyber::rhi::ShaderBytecodeFormat::HlslSource;
    shaderDesc.code = AsBytes(kLightCullingShader, std::strlen(kLightCullingShader));
    shaderDesc.entryPoint = "main";
    shaderDesc.label = "gpu cluster light culling";
    auto shader = device.CreateShader(shaderDesc);
    if (!shader)
    {
        return ResultType::Err(shader.Error());
    }
    impl.shader = shader.Value();
    cyber::rhi::PipelineLayoutDesc layoutDesc{};
    layoutDesc.groups[static_cast<u32>(cyber::rhi::BindGroupSlot::PerFrame)] = impl.groupLayout;
    auto layout = device.CreatePipelineLayout(layoutDesc);
    if (!layout)
    {
        return ResultType::Err(layout.Error());
    }
    impl.layout = layout.Value();
    cyber::rhi::ComputePipelineDesc pipelineDesc{};
    pipelineDesc.layout = impl.layout;
    pipelineDesc.computeShader = impl.shader;
    pipelineDesc.label = "gpu cluster light culling";
    auto pipeline = device.CreateComputePipeline(pipelineDesc);
    if (!pipeline)
    {
        return ResultType::Err(pipeline.Error());
    }
    impl.pipeline = pipeline.Value();
    return ResultType::Ok(static_cast<GpuLightCullingPass&&>(pass));
}

Status<Error> GpuLightCullingPass::RecordLightUpload(cyber::rhi::CommandList& list,
                                                     Span<const ClusterLightBound> lights) noexcept
{
    if (m_impl == nullptr)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu light culling: the pass is empty");
    }
    Impl& impl = *m_impl;
    if (lights.Size() > impl.desc.maxLights)
    {
        return Fail(ErrorCode::OutOfRange, "gpu light culling: more lights than the capacity");
    }
    const ClusterGridDesc& grid = impl.desc.grid;
    LightCullParams params{};
    params.grid0[0] = grid.widthPixels;
    params.grid0[1] = grid.heightPixels;
    params.grid0[2] = grid.tileSizePixels;
    params.grid0[3] = grid.depthSlices;
    params.grid1[0] = impl.tilesX;
    params.grid1[1] = impl.tilesY;
    params.grid1[2] = static_cast<u32>(lights.Size());
    params.grid1[3] = impl.desc.maxLightsPerCluster;
    params.projection[0] = std::tan(grid.verticalFovRadians * 0.5f);
    params.projection[1] = grid.nearPlane;
    params.projection[2] = grid.farPlane;
    params.projection[3] = static_cast<f32>(grid.widthPixels) / static_cast<f32>(grid.heightPixels);

    cyber::rhi::Device& device = *impl.device;
    static_assert(sizeof(ClusterLightBound) == sizeof(f32) * 4, "a light bound is one float4");
    Status<Error> step = device.WriteBuffer(impl.staging, 0, AsBytes(&params, sizeof(params)));
    if (step && !lights.Empty())
    {
        step =
            device.WriteBuffer(impl.staging, kParamBytes,
                               AsBytes(lights.Data(), lights.Size() * sizeof(ClusterLightBound)));
    }
    if (step)
    {
        step = impl.PrepareStaging(list);
    }
    if (step)
    {
        step = impl.Move(list, impl.params, impl.paramsState, ResourceState::CopyDestination);
    }
    if (step)
    {
        step = impl.Move(list, impl.lights, impl.lightsState, ResourceState::CopyDestination);
    }
    if (step)
    {
        step = list.CopyBuffer(
            cyber::rhi::BufferCopy{impl.staging, 0, impl.params, 0, sizeof(params)});
    }
    if (step && !lights.Empty())
    {
        step = list.CopyBuffer(cyber::rhi::BufferCopy{impl.staging, kParamBytes, impl.lights, 0,
                                                      lights.Size() * sizeof(ClusterLightBound)});
    }
    if (step)
    {
        step = impl.Move(list, impl.params, impl.paramsState, ResourceState::ConstantBuffer);
    }
    if (step)
    {
        step = impl.Move(list, impl.lights, impl.lightsState, ResourceState::ShaderRead);
    }
    if (step)
    {
        impl.lightCount = static_cast<u32>(lights.Size());
        impl.lightsUploaded = true;
    }
    return step;
}

Status<Error> GpuLightCullingPass::RecordCull(cyber::rhi::CommandList& list) noexcept
{
    if (m_impl == nullptr)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu light culling: the pass is empty");
    }
    Impl& impl = *m_impl;
    if (!impl.lightsUploaded)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu light culling: no lights have been uploaded");
    }
    Status<Error> step =
        impl.Move(list, impl.headers, impl.headersState, ResourceState::ShaderWrite);
    if (step)
    {
        step = impl.Move(list, impl.indices, impl.indicesState, ResourceState::ShaderWrite);
    }
    if (step)
    {
        step = list.SetPipeline(impl.pipeline);
    }
    if (step)
    {
        step = list.SetBindGroup(cyber::rhi::BindGroupSlot::PerFrame, impl.group);
    }
    if (step)
    {
        cyber::rhi::DispatchArguments dispatch{};
        dispatch.groupCountX = (impl.clusterCount + kThreadsPerGroup - 1) / kThreadsPerGroup;
        step = list.Dispatch(dispatch);
    }
    return step;
}

Status<Error> GpuLightCullingPass::RecordReadback(cyber::rhi::CommandList& list) noexcept
{
    if (m_impl == nullptr)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu light culling: the pass is empty");
    }
    Impl& impl = *m_impl;
    const u64 headerBytes = static_cast<u64>(impl.clusterCount) * sizeof(GpuClusterHeader);
    const u64 indexBytes =
        static_cast<u64>(impl.clusterCount) * impl.desc.maxLightsPerCluster * sizeof(u32);
    Status<Error> step = impl.PrepareStaging(list);
    if (step)
    {
        step = impl.Move(list, impl.headers, impl.headersState, ResourceState::CopySource);
    }
    if (step)
    {
        step = impl.Move(list, impl.indices, impl.indicesState, ResourceState::CopySource);
    }
    if (step)
    {
        step = list.CopyBuffer(
            cyber::rhi::BufferCopy{impl.headers, 0, impl.headerReadback, 0, headerBytes});
    }
    if (step)
    {
        step = list.CopyBuffer(
            cyber::rhi::BufferCopy{impl.indices, 0, impl.indexReadback, 0, indexBytes});
    }
    return step;
}

Status<Error> GpuLightCullingPass::ReadBack(cyber::memory::DynamicArray<GpuClusterHeader>& headers,
                                            cyber::memory::DynamicArray<u32>& indices) noexcept
{
    if (m_impl == nullptr)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu light culling: the pass is empty");
    }
    Impl& impl = *m_impl;
    const usize indexCount = static_cast<usize>(impl.clusterCount) * impl.desc.maxLightsPerCluster;
    if (!headers.TryResize(impl.clusterCount) || !indices.TryResize(indexCount))
    {
        return Fail(ErrorCode::Internal, "gpu light culling: readback allocation failed");
    }
    Status<Error> step = impl.device->ReadBuffer(
        impl.headerReadback, 0,
        cyber::memory::Span<Byte>(reinterpret_cast<Byte*>(headers.Data()),
                                  headers.Size() * sizeof(GpuClusterHeader)));
    if (step)
    {
        step = impl.device->ReadBuffer(
            impl.indexReadback, 0,
            cyber::memory::Span<Byte>(reinterpret_cast<Byte*>(indices.Data()),
                                      indices.Size() * sizeof(u32)));
    }
    return step;
}

bool GpuLightCullingPass::IsValid() const noexcept
{
    return m_impl != nullptr;
}

u32 GpuLightCullingPass::ClusterCount() const noexcept
{
    return m_impl != nullptr ? m_impl->clusterCount : 0;
}

u32 GpuLightCullingPass::MaxLightsPerCluster() const noexcept
{
    return m_impl != nullptr ? m_impl->desc.maxLightsPerCluster : 0;
}

BufferHandle GpuLightCullingPass::HeaderBuffer() const noexcept
{
    return m_impl != nullptr ? m_impl->headers : BufferHandle{};
}

BufferHandle GpuLightCullingPass::IndexBuffer() const noexcept
{
    return m_impl != nullptr ? m_impl->indices : BufferHandle{};
}

}  // namespace cyber::renderer
