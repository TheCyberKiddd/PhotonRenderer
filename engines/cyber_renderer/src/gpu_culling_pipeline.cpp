#include <cyber/memory/dynamic_array.h>
#include <cyber/renderer/gpu_culling_pipeline.h>
#include <cyber/rhi/binding.h>
#include <cyber/rhi/buffer.h>
#include <cyber/rhi/capabilities.h>
#include <cyber/rhi/pipeline.h>
#include <cyber/rhi/shader.h>

#include <cstring>
#include <initializer_list>
#include <new>

namespace cyber::renderer
{

namespace
{

using cyber::core::ErrorCode;
using cyber::core::usize;
using cyber::memory::Byte;
using cyber::memory::Span;
using cyber::rhi::BindGroupHandle;
using cyber::rhi::BindGroupLayoutEntry;
using cyber::rhi::BindGroupLayoutHandle;
using cyber::rhi::BindingResource;
using cyber::rhi::BindingType;
using cyber::rhi::BufferBarrier;
using cyber::rhi::BufferHandle;
using cyber::rhi::BufferUsage;
using cyber::rhi::CommandList;
using cyber::rhi::PipelineHandle;
using cyber::rhi::PipelineLayoutHandle;
using cyber::rhi::ResourceState;
using cyber::rhi::ShaderHandle;
using cyber::rhi::ShaderStage;

// ---------------------------------------------------------------------------
// Shaders
//
// Every comparison that decides visibility is written in the order the CPU
// reference evaluates it, and `precise` stops the compiler from fusing or
// reassociating it. Additions, multiplications, and comparisons then round
// identically on both sides; division and square root are within the API's
// ULP bounds rather than exact, which is what the decision margins in
// `gpu_driven.h` measure.
// ---------------------------------------------------------------------------

constexpr char kCullConstantsHlsl[] = R"HLSL(
cbuffer CullConstants : register(b0, space0)
{
    float4 Planes[6];
    float4 OcclusionColumns[4];
    float4 CameraScale;   // xyz camera position, w projection scale in pixels
    float4 NearPad;       // x near distance
    uint4 Counts;         // x instances, y batches, z flags, w groups along X
    uint4 HiZInfo;        // x width, y height, z levels, w visible capacity
};

struct DrawBatch
{
    uint indexCount;
    uint firstIndex;
    int vertexOffset;
    uint visibleBase;
    uint capacity;
    uint3 reserved;
};
)HLSL";

constexpr char kCullShaderBody[] = R"HLSL(
struct Instance
{
    float3 boundsMin;
    uint drawGroup;
    float3 boundsMax;
    uint instanceId;
};

struct DrawGroup
{
    uint lodCount;
    uint firstBatch;
    uint instanceCount;
    uint reserved;
    float4 thresholds;
};

StructuredBuffer<Instance> Instances : register(t0, space0);
StructuredBuffer<DrawGroup> Groups : register(t1, space0);
StructuredBuffer<DrawBatch> Batches : register(t2, space0);
StructuredBuffer<float> HiZ : register(t3, space0);
RWStructuredBuffer<uint> BatchCounts : register(u0, space0);
RWStructuredBuffer<uint> Visible : register(u1, space0);
RWStructuredBuffer<uint> Counters : register(u2, space0);

static const float kFloatMax = 3.402823466e+38;
static const float kNearGuard = 1.0e-5;
static const uint kHiZMaxLevels = 16;

groupshared uint g_frustumCulled;
groupshared uint g_occlusionCulled;
groupshared uint g_visible;
groupshared uint g_overflow;

// A level's width, height, and first texel, from level 0's size: each level
// halves the one below with ceiling division and follows it in the buffer.
// Derived rather than read from a table so the culling pass indexes no
// constant-buffer array by a computed value.
uint3 HiZShape(uint level)
{
    uint width = HiZInfo.x;
    uint height = HiZInfo.y;
    uint offset = 0;
    for (uint l = 0; l < level; ++l)
    {
        offset += width * height;
        width = (width + 1) / 2;
        height = (height + 1) / 2;
    }
    return uint3(width, height, offset);
}

float HiZLoad(uint3 shape, uint x, uint y)
{
    uint cx = min(x, shape.x - 1);
    uint cy = min(y, shape.y - 1);
    return HiZ[shape.z + cy * shape.x + cx];
}

uint ClampTexel(float value, uint extent)
{
    float floored = floor(value);
    if (floored <= 0.0)
        return 0;
    return min((uint)floored, extent - 1);
}

bool Occluded(float3 boundsMin, float3 boundsMax)
{
    precise float minX = kFloatMax;
    precise float minY = kFloatMax;
    precise float maxX = -kFloatMax;
    precise float maxY = -kFloatMax;
    precise float maxZ = -kFloatMax;
    precise float minW = kFloatMax;
    [unroll]
    for (uint corner = 0; corner < 8; ++corner)
    {
        float px = (corner & 1) != 0 ? boundsMax.x : boundsMin.x;
        float py = (corner & 2) != 0 ? boundsMax.y : boundsMin.y;
        float pz = (corner & 4) != 0 ? boundsMax.z : boundsMin.z;
        precise float4 clip = ((OcclusionColumns[0] * px + OcclusionColumns[1] * py) +
                               OcclusionColumns[2] * pz) + OcclusionColumns[3];
        minW = clip.w < minW ? clip.w : minW;
        if (clip.w > kNearGuard)
        {
            precise float x = clip.x / clip.w;
            precise float y = clip.y / clip.w;
            precise float z = clip.z / clip.w;
            minX = x < minX ? x : minX;
            maxX = x > maxX ? x : maxX;
            minY = y < minY ? y : minY;
            maxY = y > maxY ? y : maxY;
            maxZ = z > maxZ ? z : maxZ;
        }
    }
    if (minW <= kNearGuard)
        return false;
    if (maxX < -1.0 || minX > 1.0 || maxY < -1.0 || minY > 1.0)
        return false;
    minX = minX > -1.0 ? minX : -1.0;
    maxX = maxX < 1.0 ? maxX : 1.0;
    minY = minY > -1.0 ? minY : -1.0;
    maxY = maxY < 1.0 ? maxY : 1.0;

    float width = (float)HiZInfo.x;
    float height = (float)HiZInfo.y;
    precise float left = (minX * 0.5 + 0.5) * width;
    precise float right = (maxX * 0.5 + 0.5) * width;
    precise float top = (0.5 - maxY * 0.5) * height;
    precise float bottom = (0.5 - minY * 0.5) * height;
    uint x0 = ClampTexel(left, HiZInfo.x);
    uint x1 = ClampTexel(right, HiZInfo.x);
    uint y0 = ClampTexel(top, HiZInfo.y);
    uint y1 = ClampTexel(bottom, HiZInfo.y);

    // The finest level at which the rectangle spans at most two texels per
    // axis. Coordinates step down one level at a time with constant one-bit
    // shifts, which is `x >> level` without a shift by a computed amount:
    // WARP was measured selecting a coarser level than the shifted-by-level
    // form of this loop allows, and this form agrees on both adapters.
    uint level = 0;
    uint lx0 = x0;
    uint lx1 = x1;
    uint ly0 = y0;
    uint ly1 = y1;
    [loop]
    for (uint step = 0; step + 1 < kHiZMaxLevels; ++step)
    {
        if (level + 1 >= HiZInfo.z || (lx1 - lx0 <= 1 && ly1 - ly0 <= 1))
            break;
        lx0 >>= 1;
        lx1 >>= 1;
        ly0 >>= 1;
        ly1 >>= 1;
        ++level;
    }
    uint3 shape = HiZShape(level);
    float a = HiZLoad(shape, lx0, ly0);
    float b = HiZLoad(shape, lx1, ly0);
    float c = HiZLoad(shape, lx0, ly1);
    float d = HiZLoad(shape, lx1, ly1);
    float ab = b < a ? b : a;
    float cd = d < c ? d : c;
    float occluder = cd < ab ? cd : ab;
    return maxZ < occluder;
}

[numthreads(64, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    if (groupIndex == 0)
    {
        g_frustumCulled = 0;
        g_occlusionCulled = 0;
        g_visible = 0;
        g_overflow = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint index = (groupId.y * Counts.w + groupId.x) * 64 + groupIndex;
    if (index < Counts.x)
    {
        Instance instance = Instances[index];
        float3 boundsMin = instance.boundsMin;
        float3 boundsMax = instance.boundsMax;
        bool visible = true;
        uint culledBy = 0;

        if ((Counts.z & 1) != 0)
        {
            for (uint planeIndex = 0; planeIndex < 6; ++planeIndex)
            {
                float4 plane = Planes[planeIndex];
                float px = plane.x >= 0.0 ? boundsMax.x : boundsMin.x;
                float py = plane.y >= 0.0 ? boundsMax.y : boundsMin.y;
                float pz = plane.z >= 0.0 ? boundsMax.z : boundsMin.z;
                precise float distance = ((plane.x * px + plane.y * py) + plane.z * pz) + plane.w;
                if (distance < 0.0)
                {
                    visible = false;
                    culledBy = 1;
                    break;
                }
            }
        }

        DrawGroup group = Groups[instance.drawGroup];
        uint lod = 0;
        if (visible && (Counts.z & 4) != 0 && group.lodCount > 1)
        {
            precise float3 center = (boundsMin + boundsMax) * 0.5;
            precise float3 extents = (boundsMax - boundsMin) * 0.5;
            precise float radius =
                sqrt((extents.x * extents.x + extents.y * extents.y) + extents.z * extents.z);
            precise float3 offset = center - CameraScale.xyz;
            precise float distance =
                sqrt((offset.x * offset.x + offset.y * offset.y) + offset.z * offset.z);
            precise float nearest = distance - radius;
            precise float denominator = NearPad.x > nearest ? NearPad.x : nearest;
            precise float projected = radius * CameraScale.w / denominator;
            float thresholds[4] = {group.thresholds.x, group.thresholds.y, group.thresholds.z,
                                   group.thresholds.w};
            for (uint l = 0; l + 1 < group.lodCount; ++l)
            {
                if (projected < thresholds[l])
                    lod = l + 1;
            }
        }

        if (visible && (Counts.z & 2) != 0 && Occluded(boundsMin, boundsMax))
        {
            visible = false;
            culledBy = 2;
        }

        if (visible)
        {
            uint batchIndex = group.firstBatch + lod;
            uint slot;
            InterlockedAdd(BatchCounts[batchIndex], 1, slot);
            DrawBatch batch = Batches[batchIndex];
            if (slot < batch.capacity)
                Visible[batch.visibleBase + slot] = instance.instanceId;
            else
                InterlockedAdd(g_overflow, 1);
            InterlockedAdd(g_visible, 1);
        }
        else if (culledBy == 1)
        {
            InterlockedAdd(g_frustumCulled, 1);
        }
        else
        {
            InterlockedAdd(g_occlusionCulled, 1);
        }
    }

    // One global atomic per counter per group rather than per instance: a
    // million instances on one address would serialize on it.
    GroupMemoryBarrierWithGroupSync();
    if (groupIndex == 0)
    {
        if (g_frustumCulled != 0)
            InterlockedAdd(Counters[1], g_frustumCulled);
        if (g_occlusionCulled != 0)
            InterlockedAdd(Counters[2], g_occlusionCulled);
        if (g_visible != 0)
            InterlockedAdd(Counters[3], g_visible);
        if (g_overflow != 0)
            InterlockedAdd(Counters[4], g_overflow);
    }
}
)HLSL";

constexpr char kCompactShaderBody[] = R"HLSL(
StructuredBuffer<DrawBatch> Batches : register(t0, space0);
RWStructuredBuffer<uint> BatchCounts : register(u0, space0);
RWStructuredBuffer<uint> Arguments : register(u1, space0);
RWStructuredBuffer<uint> Counters : register(u2, space0);

groupshared uint g_sums[256];

// One group. Each thread owns a contiguous run of batches, counts its
// non-empty ones, and a Hillis-Steele scan turns the counts into write
// positions, so draws come out in batch order: deterministic, and identical
// to the CPU reference rather than merely equivalent to it.
[numthreads(256, 1, 1)]
void main(uint thread : SV_GroupIndex)
{
    uint batchCount = Counts.y;
    uint perThread = (batchCount + 255) / 256;
    uint begin = min(thread * perThread, batchCount);
    uint end = min(begin + perThread, batchCount);
    uint local = 0;
    for (uint b = begin; b < end; ++b)
    {
        if (BatchCounts[b] != 0)
            ++local;
    }
    g_sums[thread] = local;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint offset = 1; offset < 256; offset <<= 1)
    {
        uint add = thread >= offset ? g_sums[thread - offset] : 0;
        GroupMemoryBarrierWithGroupSync();
        g_sums[thread] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint write = g_sums[thread] - local;
    for (uint batchIndex = begin; batchIndex < end; ++batchIndex)
    {
        uint count = BatchCounts[batchIndex];
        if (count == 0)
            continue;
        DrawBatch batch = Batches[batchIndex];
        uint base = write * 5;
        Arguments[base + 0] = batch.indexCount;
        Arguments[base + 1] = min(count, batch.capacity);
        Arguments[base + 2] = batch.firstIndex;
        Arguments[base + 3] = asuint(batch.vertexOffset);
        Arguments[base + 4] = batch.visibleBase;
        ++write;
    }
    if (thread == 255)
        Counters[0] = g_sums[255];
}
)HLSL";

constexpr char kHiZShader[] = R"HLSL(
cbuffer HiZPass : register(b0, space0)
{
    uint4 Source;       // x first texel, y width, z height, w row pitch
    uint4 Destination;  // x first texel, y width, z height, w 0 = copy depth, 1 = reduce
};

StructuredBuffer<float> Depth : register(t0, space0);
RWStructuredBuffer<float> Pyramid : register(u0, space0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Destination.y || id.y >= Destination.z)
        return;
    float value;
    if (Destination.w == 0)
    {
        value = Depth[Source.x + id.y * Source.w + id.x];
    }
    else
    {
        // Reverse-Z: the farthest depth is the smallest. Ceiling sizes keep
        // 2x inside the source; 2x + 1 is clamped.
        uint x0 = id.x * 2;
        uint y0 = id.y * 2;
        uint x1 = min(x0 + 1, Source.y - 1);
        uint y1 = min(y0 + 1, Source.z - 1);
        float a = Pyramid[Source.x + y0 * Source.y + x0];
        float b = Pyramid[Source.x + y0 * Source.y + x1];
        float c = Pyramid[Source.x + y1 * Source.y + x0];
        float d = Pyramid[Source.x + y1 * Source.y + x1];
        float ab = b < a ? b : a;
        float cd = d < c ? d : c;
        value = cd < ab ? cd : ab;
    }
    Pyramid[Destination.x + id.y * Destination.y + id.x] = value;
}
)HLSL";

// The constant block both culling shaders read; mirrors `CullConstants`.
struct alignas(16) GpuCullConstants
{
    f32 planes[6][4];
    f32 occlusionColumns[4][4];
    f32 cameraScale[4];
    f32 nearPad[4];
    u32 counts[4];
    u32 hiZInfo[4];
};
static_assert(sizeof(GpuCullConstants) == 224, "GpuCullConstants mirrors the HLSL cbuffer");

// Constant buffers are bound in 256-byte units.
inline constexpr u64 kConstantSlotBytes = 256;
static_assert(sizeof(GpuCullConstants) <= kConstantSlotBytes, "the constants fit one slot");

struct HiZPassConstants
{
    u32 source[4];
    u32 destination[4];
};
static_assert(sizeof(HiZPassConstants) == 32, "HiZPassConstants mirrors the HLSL cbuffer");

// Groups along X for the culling dispatch before it spills into Y.
inline constexpr u32 kMaxGroupsX = 32768;

// Bind groups kept for distinct depth sources of `RecordHiZBuild`. A group's
// descriptors may still be read by work in flight, so none is destroyed
// before the pipeline is.
inline constexpr u32 kMaxHiZSources = 4;

enum BufferId : u32
{
    kConstants = 0,
    kInstances,
    kGroups,
    kBatches,
    kHiZ,
    kBatchCounts,
    kVisible,
    kArguments,
    kCounters,
    kTrackedBufferCount
};

[[nodiscard]] Status<Error> Fail(ErrorCode code, const char* message) noexcept
{
    return Status<Error>::Err(Error{code, message});
}

[[nodiscard]] Span<const Byte> AsBytes(const void* data, usize size) noexcept
{
    return Span<const Byte>(static_cast<const Byte*>(data), size);
}

[[nodiscard]] BindGroupLayoutEntry Entry(u32 binding, BindingType type, u32 shaderRegister) noexcept
{
    BindGroupLayoutEntry entry{};
    entry.binding = binding;
    entry.type = type;
    entry.visibility = cyber::rhi::ShaderStageBit(ShaderStage::Compute);
    entry.shaderRegister = shaderRegister;
    return entry;
}

[[nodiscard]] BindingResource Resource(BindingType type, BufferHandle buffer) noexcept
{
    BindingResource resource{};
    resource.type = type;
    resource.buffer = buffer;
    return resource;
}

}  // namespace

struct GpuCullingPipeline::Impl
{
    explicit Impl(cyber::rhi::Device& owner, cyber::memory::Allocator& allocator) noexcept
        : device(&owner), allocator(&allocator), shaderText(allocator)
    {
    }

    cyber::rhi::Device* device = nullptr;
    cyber::memory::Allocator* allocator = nullptr;
    GpuCullingPipelineDesc desc{};
    cyber::memory::DynamicArray<char> shaderText;

    BufferHandle buffers[kTrackedBufferCount]{};
    ResourceState states[kTrackedBufferCount]{};
    BufferHandle constantStaging{};
    BufferHandle sceneStaging{};
    BufferHandle hiZStaging{};
    BufferHandle zeroSource{};
    BufferHandle counterReadback{};
    BufferHandle batchCountReadback{};
    BufferHandle visibleReadback{};
    BufferHandle argumentReadback{};
    BufferHandle hiZReadback{};
    bool stagingReady = false;

    ShaderHandle cullShader{};
    ShaderHandle compactShader{};
    ShaderHandle hiZShader{};
    BindGroupLayoutHandle cullGroupLayout{};
    BindGroupLayoutHandle compactGroupLayout{};
    BindGroupLayoutHandle hiZGroupLayout{};
    PipelineLayoutHandle cullLayout{};
    PipelineLayoutHandle compactLayout{};
    PipelineLayoutHandle hiZLayout{};
    PipelineHandle cullPipeline{};
    PipelineHandle compactPipeline{};
    PipelineHandle hiZPipeline{};
    BindGroupHandle cullGroup{};
    BindGroupHandle compactGroup{};
    BufferHandle hiZSources[kMaxHiZSources]{};
    BindGroupHandle hiZGroups[kMaxHiZSources]{};
    u32 hiZSourceCount = 0;

    u64 sceneStagingBytes = 0;
    u64 hiZBytes = 0;
    u32 constantSlot = 0;
    u32 instanceCount = 0;
    u32 batchCount = 0;
    u32 visibleCapacity = 0;
    bool sceneUploaded = false;
    HiZLevel hiZLevels[kHiZMaxLevels]{};
    u32 hiZLevelCount = 0;
    bool hiZReady = false;
    bool hiZReadbackRecorded = false;
    GpuCullingPipelineStats stats{};

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

    // Compiles `prefix` followed by `body` as one compute shader.
    [[nodiscard]] Status<Error> MakeShader(ShaderHandle& out, const char* prefix, const char* body,
                                           const char* label) noexcept
    {
        const usize prefixBytes = prefix != nullptr ? std::strlen(prefix) : 0;
        const usize bodyBytes = std::strlen(body);
        if (!shaderText.TryResize(prefixBytes + bodyBytes + 1))
        {
            return Fail(ErrorCode::Internal, "gpu culling: shader text allocation failed");
        }
        if (prefixBytes > 0)
        {
            std::memcpy(shaderText.Data(), prefix, prefixBytes);
        }
        std::memcpy(shaderText.Data() + prefixBytes, body, bodyBytes);
        shaderText[prefixBytes + bodyBytes] = '\0';
        cyber::rhi::ShaderDesc shaderDesc{};
        shaderDesc.stage = ShaderStage::Compute;
        shaderDesc.format = cyber::rhi::ShaderBytecodeFormat::HlslSource;
        shaderDesc.code = AsBytes(shaderText.Data(), prefixBytes + bodyBytes);
        shaderDesc.entryPoint = "main";
        shaderDesc.label = label;
        auto created = device->CreateShader(shaderDesc);
        if (!created)
        {
            return Status<Error>::Err(created.Error());
        }
        out = created.Value();
        return Status<Error>::Ok();
    }

    [[nodiscard]] Status<Error> MakeComputePipeline(BindGroupLayoutHandle groupLayout,
                                                    ShaderHandle shader, u32 pushConstantBytes,
                                                    PipelineLayoutHandle& layoutOut,
                                                    PipelineHandle& pipelineOut,
                                                    const char* label) noexcept
    {
        cyber::rhi::PipelineLayoutDesc layoutDesc{};
        layoutDesc.groups[static_cast<u32>(cyber::rhi::BindGroupSlot::PerFrame)] = groupLayout;
        cyber::rhi::PushConstantRange range{};
        range.offsetBytes = 0;
        range.sizeBytes = pushConstantBytes;
        range.stages = cyber::rhi::ShaderStageBit(ShaderStage::Compute);
        if (pushConstantBytes > 0)
        {
            layoutDesc.pushConstants = Span<const cyber::rhi::PushConstantRange>(&range, 1);
        }
        layoutDesc.label = label;
        auto layout = device->CreatePipelineLayout(layoutDesc);
        if (!layout)
        {
            return Status<Error>::Err(layout.Error());
        }
        layoutOut = layout.Value();
        cyber::rhi::ComputePipelineDesc pipelineDesc{};
        pipelineDesc.layout = layoutOut;
        pipelineDesc.computeShader = shader;
        pipelineDesc.label = label;
        auto pipeline = device->CreateComputePipeline(pipelineDesc);
        if (!pipeline)
        {
            return Status<Error>::Err(pipeline.Error());
        }
        pipelineOut = pipeline.Value();
        return Status<Error>::Ok();
    }

    [[nodiscard]] Status<Error> MakeGroupLayout(Span<const BindGroupLayoutEntry> entries,
                                                BindGroupLayoutHandle& out) noexcept
    {
        cyber::rhi::BindGroupLayoutDesc layoutDesc{};
        layoutDesc.entries = entries;
        auto created = device->CreateBindGroupLayout(layoutDesc);
        if (!created)
        {
            return Status<Error>::Err(created.Error());
        }
        out = created.Value();
        return Status<Error>::Ok();
    }

    [[nodiscard]] Status<Error> MakeGroup(BindGroupLayoutHandle layout,
                                          Span<const BindingResource> resources,
                                          BindGroupHandle& out) noexcept
    {
        cyber::rhi::BindGroupDesc groupDesc{};
        groupDesc.layout = layout;
        groupDesc.resources = resources;
        auto created = device->CreateBindGroup(groupDesc);
        if (!created)
        {
            return Status<Error>::Err(created.Error());
        }
        out = created.Value();
        return Status<Error>::Ok();
    }

    // Moves tracked buffers to `to`. A buffer already in `to` records nothing,
    // except that `ShaderWrite` to `ShaderWrite` records a UAV barrier: two
    // dispatches writing one buffer still need their writes ordered.
    [[nodiscard]] Status<Error> Transition(CommandList& list, Span<const BufferId> ids,
                                           ResourceState to) noexcept
    {
        BufferBarrier barriers[kTrackedBufferCount]{};
        u32 count = 0;
        for (usize index = 0; index < ids.Size(); ++index)
        {
            const BufferId id = ids[index];
            if (states[id] == to && to != ResourceState::ShaderWrite)
            {
                continue;
            }
            barriers[count++] = BufferBarrier{buffers[id], states[id], to};
        }
        if (count == 0)
        {
            return Status<Error>::Ok();
        }
        const Status<Error> recorded = list.Barrier(Span<const BufferBarrier>(barriers, count));
        if (!recorded)
        {
            return recorded;
        }
        for (usize index = 0; index < ids.Size(); ++index)
        {
            states[ids[index]] = to;
        }
        return Status<Error>::Ok();
    }

    [[nodiscard]] Status<Error> Transition(CommandList& list, std::initializer_list<BufferId> ids,
                                           ResourceState to) noexcept
    {
        return Transition(list, Span<const BufferId>(ids.begin(), ids.size()), to);
    }

    // Staging, zero, and readback buffers live in heaps whose state is fixed;
    // the RHI still tracks the state a copy names, so each is moved once.
    [[nodiscard]] Status<Error> PrepareStaging(CommandList& list) noexcept
    {
        if (stagingReady)
        {
            return Status<Error>::Ok();
        }
        const BufferBarrier barriers[] = {
            {constantStaging, ResourceState::Undefined, ResourceState::CopySource},
            {sceneStaging, ResourceState::Undefined, ResourceState::CopySource},
            {hiZStaging, ResourceState::Undefined, ResourceState::CopySource},
            {zeroSource, ResourceState::Undefined, ResourceState::CopySource},
            {counterReadback, ResourceState::Undefined, ResourceState::CopyDestination},
            {batchCountReadback, ResourceState::Undefined, ResourceState::CopyDestination},
            {visibleReadback, ResourceState::Undefined, ResourceState::CopyDestination},
            {argumentReadback, ResourceState::Undefined, ResourceState::CopyDestination},
            {hiZReadback, ResourceState::Undefined, ResourceState::CopyDestination},
        };
        const Status<Error> recorded = list.Barrier(
            Span<const BufferBarrier>(barriers, sizeof(barriers) / sizeof(barriers[0])));
        if (!recorded)
        {
            return recorded;
        }
        stagingReady = true;
        return Status<Error>::Ok();
    }

    [[nodiscard]] Status<Error> Copy(CommandList& list, BufferHandle source, u64 sourceOffset,
                                     BufferHandle destination, u64 destinationOffset,
                                     u64 bytes) noexcept
    {
        if (bytes == 0)
        {
            return Status<Error>::Ok();
        }
        cyber::rhi::BufferCopy copy{};
        copy.source = source;
        copy.sourceOffset = sourceOffset;
        copy.destination = destination;
        copy.destinationOffset = destinationOffset;
        copy.sizeBytes = bytes;
        return list.CopyBuffer(copy);
    }

    void DestroyAll() noexcept
    {
        cyber::rhi::Device& rhi = *device;
        for (u32 index = 0; index < hiZSourceCount; ++index)
        {
            (void)rhi.DestroyBindGroup(hiZGroups[index]);
        }
        const BindGroupHandle groups[] = {cullGroup, compactGroup};
        for (const BindGroupHandle group : groups)
        {
            if (group.IsValid())
            {
                (void)rhi.DestroyBindGroup(group);
            }
        }
        const PipelineHandle pipelines[] = {cullPipeline, compactPipeline, hiZPipeline};
        for (const PipelineHandle pipeline : pipelines)
        {
            if (pipeline.IsValid())
            {
                (void)rhi.DestroyComputePipeline(pipeline);
            }
        }
        const PipelineLayoutHandle layouts[] = {cullLayout, compactLayout, hiZLayout};
        for (const PipelineLayoutHandle layout : layouts)
        {
            if (layout.IsValid())
            {
                (void)rhi.DestroyPipelineLayout(layout);
            }
        }
        const BindGroupLayoutHandle groupLayouts[] = {cullGroupLayout, compactGroupLayout,
                                                      hiZGroupLayout};
        for (const BindGroupLayoutHandle layout : groupLayouts)
        {
            if (layout.IsValid())
            {
                (void)rhi.DestroyBindGroupLayout(layout);
            }
        }
        const ShaderHandle shaders[] = {cullShader, compactShader, hiZShader};
        for (const ShaderHandle shader : shaders)
        {
            if (shader.IsValid())
            {
                (void)rhi.DestroyShader(shader);
            }
        }
        for (const BufferHandle buffer : buffers)
        {
            if (buffer.IsValid())
            {
                (void)rhi.DestroyBuffer(buffer);
            }
        }
        const BufferHandle extra[] = {constantStaging, sceneStaging,     hiZStaging,
                                      zeroSource,      counterReadback,  batchCountReadback,
                                      visibleReadback, argumentReadback, hiZReadback};
        for (const BufferHandle buffer : extra)
        {
            if (buffer.IsValid())
            {
                (void)rhi.DestroyBuffer(buffer);
            }
        }
    }
};

GpuCullingPipeline::GpuCullingPipeline(Impl* impl) noexcept : m_impl(impl) {}

GpuCullingPipeline::~GpuCullingPipeline()
{
    Destroy();
}

GpuCullingPipeline::GpuCullingPipeline(GpuCullingPipeline&& other) noexcept : m_impl(other.m_impl)
{
    other.m_impl = nullptr;
}

GpuCullingPipeline& GpuCullingPipeline::operator=(GpuCullingPipeline&& other) noexcept
{
    if (this != &other)
    {
        Destroy();
        m_impl = other.m_impl;
        other.m_impl = nullptr;
    }
    return *this;
}

void GpuCullingPipeline::Destroy() noexcept
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

Result<GpuCullingPipeline, Error>
GpuCullingPipeline::Create(cyber::rhi::Device& device, const GpuCullingPipelineDesc& desc,
                           cyber::memory::Allocator& allocator) noexcept
{
    using ResultType = Result<GpuCullingPipeline, Error>;
    const cyber::rhi::Capabilities capabilities = device.Capabilities();
    if (!capabilities.features.Has(cyber::rhi::Feature::Compute) ||
        !capabilities.features.Has(cyber::rhi::Feature::IndirectDrawCount))
    {
        return ResultType::Err(Error{ErrorCode::Unsupported,
                                     "gpu culling: the device needs compute and indirect-count "
                                     "draws"});
    }
    if (desc.maxBatches == 0 || desc.maxGroups == 0)
    {
        return ResultType::Err(
            Error{ErrorCode::InvalidArgument, "gpu culling: a pipeline needs batch capacity"});
    }
    HiZLevel levels[kHiZMaxLevels]{};
    u32 hiZTexels = 1;
    if (desc.hiZMaxWidth != 0 || desc.hiZMaxHeight != 0)
    {
        if (ComputeHiZLayout(desc.hiZMaxWidth, desc.hiZMaxHeight, levels, &hiZTexels) == 0)
        {
            return ResultType::Err(
                Error{ErrorCode::InvalidArgument, "gpu culling: invalid Hi-Z capacity"});
        }
    }

    void* memory = allocator.Allocate(sizeof(Impl), alignof(Impl));
    if (memory == nullptr)
    {
        return ResultType::Err(Error{ErrorCode::Internal, "gpu culling: allocation failed"});
    }
    GpuCullingPipeline pipeline(new (memory) Impl(device, allocator));
    Impl& impl = *pipeline.m_impl;
    impl.desc = desc;
    impl.hiZBytes = static_cast<u64>(hiZTexels) * sizeof(f32);

    const u64 instanceBytes = static_cast<u64>(desc.maxInstances) * sizeof(GpuCullInstance);
    const u64 groupBytes = static_cast<u64>(desc.maxGroups) * sizeof(GpuDrawGroup);
    const u64 batchBytes = static_cast<u64>(desc.maxBatches) * sizeof(GpuDrawBatch);
    const u64 batchCountBytes = static_cast<u64>(desc.maxBatches) * sizeof(u32);
    const u64 visibleBytes = static_cast<u64>(desc.maxVisible) * sizeof(u32);
    const u64 argumentBytes =
        static_cast<u64>(desc.maxBatches) * cyber::rhi::kIndirectDrawIndexedStrideBytes;
    impl.sceneStagingBytes = instanceBytes + groupBytes + batchBytes;

    const BufferUsage deviceCopy = BufferUsage::Storage | BufferUsage::CopyDestination;
    const BufferUsage staging = BufferUsage::HostVisible | BufferUsage::CopySource;
    const BufferUsage readback = BufferUsage::Readback | BufferUsage::CopyDestination;
    const Status<Error> buffers[] = {
        impl.MakeBuffer(impl.buffers[kConstants],
                        BufferUsage::Uniform | BufferUsage::CopyDestination, kConstantSlotBytes, 0,
                        "gpu cull constants"),
        impl.MakeBuffer(impl.buffers[kInstances], deviceCopy, instanceBytes,
                        sizeof(GpuCullInstance), "gpu cull instances"),
        impl.MakeBuffer(impl.buffers[kGroups], deviceCopy, groupBytes, sizeof(GpuDrawGroup),
                        "gpu cull groups"),
        impl.MakeBuffer(impl.buffers[kBatches], deviceCopy, batchBytes, sizeof(GpuDrawBatch),
                        "gpu cull batches"),
        impl.MakeBuffer(impl.buffers[kHiZ], deviceCopy | BufferUsage::CopySource, impl.hiZBytes,
                        sizeof(f32), "gpu cull hi-z"),
        impl.MakeBuffer(impl.buffers[kBatchCounts], deviceCopy | BufferUsage::CopySource,
                        batchCountBytes, sizeof(u32), "gpu cull batch counts"),
        impl.MakeBuffer(impl.buffers[kVisible],
                        BufferUsage::Storage | BufferUsage::Vertex | BufferUsage::CopySource,
                        visibleBytes, sizeof(u32), "gpu cull visible instances"),
        impl.MakeBuffer(impl.buffers[kArguments],
                        BufferUsage::Storage | BufferUsage::Indirect | BufferUsage::CopySource,
                        argumentBytes, sizeof(u32), "gpu cull draw arguments"),
        impl.MakeBuffer(impl.buffers[kCounters],
                        deviceCopy | BufferUsage::Indirect | BufferUsage::CopySource,
                        sizeof(GpuCullCounters), sizeof(u32), "gpu cull counters"),
        impl.MakeBuffer(impl.constantStaging, staging, kConstantSlotBytes * kGpuCullConstantSlots,
                        0, "gpu cull constants staging"),
        impl.MakeBuffer(impl.sceneStaging, staging, impl.sceneStagingBytes, 0,
                        "gpu cull scene staging"),
        impl.MakeBuffer(impl.hiZStaging, staging, impl.hiZBytes, 0, "gpu cull hi-z staging"),
        impl.MakeBuffer(impl.zeroSource, staging,
                        batchCountBytes > sizeof(GpuCullCounters) ? batchCountBytes
                                                                  : sizeof(GpuCullCounters),
                        0, "gpu cull zeros"),
        impl.MakeBuffer(impl.counterReadback, readback, sizeof(GpuCullCounters), 0,
                        "gpu cull counters readback"),
        impl.MakeBuffer(impl.batchCountReadback, readback, batchCountBytes, 0,
                        "gpu cull batch counts readback"),
        impl.MakeBuffer(impl.visibleReadback, readback, visibleBytes, 0,
                        "gpu cull visible readback"),
        impl.MakeBuffer(impl.argumentReadback, readback, argumentBytes, 0,
                        "gpu cull arguments readback"),
        impl.MakeBuffer(impl.hiZReadback, readback, impl.hiZBytes, 0, "gpu cull hi-z readback"),
    };
    for (const Status<Error>& status : buffers)
    {
        if (!status)
        {
            return ResultType::Err(status.Error());
        }
    }

    // The zero source is written once; every frame's clears copy from it.
    {
        const u64 zeroBytes =
            batchCountBytes > sizeof(GpuCullCounters) ? batchCountBytes : sizeof(GpuCullCounters);
        cyber::memory::DynamicArray<Byte> zeros(allocator);
        if (!zeros.TryResize(static_cast<usize>(zeroBytes)))
        {
            return ResultType::Err(Error{ErrorCode::Internal, "gpu culling: allocation failed"});
        }
        std::memset(zeros.Data(), 0, zeros.Size());
        const Status<Error> written =
            device.WriteBuffer(impl.zeroSource, 0, Span<const Byte>(zeros.Data(), zeros.Size()));
        if (!written)
        {
            return ResultType::Err(written.Error());
        }
    }

    const BindGroupLayoutEntry cullEntries[] = {
        Entry(0, BindingType::UniformBuffer, 0),
        Entry(1, BindingType::ReadOnlyStorageBuffer, 0),
        Entry(2, BindingType::ReadOnlyStorageBuffer, 1),
        Entry(3, BindingType::ReadOnlyStorageBuffer, 2),
        Entry(4, BindingType::ReadOnlyStorageBuffer, 3),
        Entry(5, BindingType::StorageBuffer, 0),
        Entry(6, BindingType::StorageBuffer, 1),
        Entry(7, BindingType::StorageBuffer, 2),
    };
    const BindGroupLayoutEntry compactEntries[] = {
        Entry(0, BindingType::UniformBuffer, 0), Entry(1, BindingType::ReadOnlyStorageBuffer, 0),
        Entry(2, BindingType::StorageBuffer, 0), Entry(3, BindingType::StorageBuffer, 1),
        Entry(4, BindingType::StorageBuffer, 2),
    };
    const BindGroupLayoutEntry hiZEntries[] = {
        Entry(0, BindingType::ReadOnlyStorageBuffer, 0),
        Entry(1, BindingType::StorageBuffer, 0),
    };
    const BindingResource cullResources[] = {
        Resource(BindingType::UniformBuffer, impl.buffers[kConstants]),
        Resource(BindingType::ReadOnlyStorageBuffer, impl.buffers[kInstances]),
        Resource(BindingType::ReadOnlyStorageBuffer, impl.buffers[kGroups]),
        Resource(BindingType::ReadOnlyStorageBuffer, impl.buffers[kBatches]),
        Resource(BindingType::ReadOnlyStorageBuffer, impl.buffers[kHiZ]),
        Resource(BindingType::StorageBuffer, impl.buffers[kBatchCounts]),
        Resource(BindingType::StorageBuffer, impl.buffers[kVisible]),
        Resource(BindingType::StorageBuffer, impl.buffers[kCounters]),
    };
    const BindingResource compactResources[] = {
        Resource(BindingType::UniformBuffer, impl.buffers[kConstants]),
        Resource(BindingType::ReadOnlyStorageBuffer, impl.buffers[kBatches]),
        Resource(BindingType::StorageBuffer, impl.buffers[kBatchCounts]),
        Resource(BindingType::StorageBuffer, impl.buffers[kArguments]),
        Resource(BindingType::StorageBuffer, impl.buffers[kCounters]),
    };
    Status<Error> step = impl.MakeGroupLayout(
        Span<const BindGroupLayoutEntry>(cullEntries, sizeof(cullEntries) / sizeof(cullEntries[0])),
        impl.cullGroupLayout);
    if (step)
    {
        step = impl.MakeGroupLayout(
            Span<const BindGroupLayoutEntry>(compactEntries,
                                             sizeof(compactEntries) / sizeof(compactEntries[0])),
            impl.compactGroupLayout);
    }
    if (step)
    {
        step = impl.MakeGroupLayout(Span<const BindGroupLayoutEntry>(
                                        hiZEntries, sizeof(hiZEntries) / sizeof(hiZEntries[0])),
                                    impl.hiZGroupLayout);
    }
    if (step)
    {
        step = impl.MakeGroup(impl.cullGroupLayout,
                              Span<const BindingResource>(
                                  cullResources, sizeof(cullResources) / sizeof(cullResources[0])),
                              impl.cullGroup);
    }
    if (step)
    {
        step = impl.MakeGroup(
            impl.compactGroupLayout,
            Span<const BindingResource>(compactResources,
                                        sizeof(compactResources) / sizeof(compactResources[0])),
            impl.compactGroup);
    }
    if (step)
    {
        step = impl.MakeShader(impl.cullShader, kCullConstantsHlsl, kCullShaderBody,
                               "gpu instance culling");
    }
    if (step)
    {
        step = impl.MakeShader(impl.compactShader, kCullConstantsHlsl, kCompactShaderBody,
                               "gpu draw compaction");
    }
    if (step)
    {
        step = impl.MakeShader(impl.hiZShader, nullptr, kHiZShader, "gpu hi-z reduction");
    }
    if (step)
    {
        step = impl.MakeComputePipeline(impl.cullGroupLayout, impl.cullShader, 0, impl.cullLayout,
                                        impl.cullPipeline, "gpu instance culling");
    }
    if (step)
    {
        step = impl.MakeComputePipeline(impl.compactGroupLayout, impl.compactShader, 0,
                                        impl.compactLayout, impl.compactPipeline,
                                        "gpu draw compaction");
    }
    if (step)
    {
        step =
            impl.MakeComputePipeline(impl.hiZGroupLayout, impl.hiZShader, sizeof(HiZPassConstants),
                                     impl.hiZLayout, impl.hiZPipeline, "gpu hi-z reduction");
    }
    if (!step)
    {
        return ResultType::Err(step.Error());
    }
    // The shader text is only needed while compiling.
    impl.shaderText.Clear();
    return ResultType::Ok(static_cast<GpuCullingPipeline&&>(pipeline));
}

Status<Error> GpuCullingPipeline::RecordSceneUpload(CommandList& list,
                                                    const GpuDrivenScene& scene) noexcept
{
    if (m_impl == nullptr)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu culling: the pipeline is empty");
    }
    Impl& impl = *m_impl;
    if (!scene.IsFinalized())
    {
        return Fail(ErrorCode::InvalidArgument, "gpu culling: the scene is not finalized");
    }
    const Span<const GpuCullInstance> instances = scene.Instances();
    const Span<const GpuDrawGroup> groups = scene.Groups();
    const Span<const GpuDrawBatch> batches = scene.Batches();
    if (instances.Size() > impl.desc.maxInstances || groups.Size() > impl.desc.maxGroups ||
        batches.Size() > impl.desc.maxBatches || scene.VisibleCapacity() > impl.desc.maxVisible)
    {
        return Fail(ErrorCode::OutOfRange,
                    "gpu culling: the scene exceeds the pipeline's capacity");
    }
    if (batches.Empty())
    {
        return Fail(ErrorCode::InvalidArgument, "gpu culling: the scene has no batches");
    }
    const u64 instanceBytes = instances.Size() * sizeof(GpuCullInstance);
    const u64 groupBytes = groups.Size() * sizeof(GpuDrawGroup);
    const u64 batchBytes = batches.Size() * sizeof(GpuDrawBatch);
    cyber::rhi::Device& device = *impl.device;
    Status<Error> step = Status<Error>::Ok();
    if (instanceBytes > 0)
    {
        step = device.WriteBuffer(impl.sceneStaging, 0, AsBytes(instances.Data(), instanceBytes));
    }
    if (step)
    {
        step = device.WriteBuffer(impl.sceneStaging, instanceBytes,
                                  AsBytes(groups.Data(), groupBytes));
    }
    if (step)
    {
        step = device.WriteBuffer(impl.sceneStaging, instanceBytes + groupBytes,
                                  AsBytes(batches.Data(), batchBytes));
    }
    if (step)
    {
        step = impl.PrepareStaging(list);
    }
    if (step)
    {
        step =
            impl.Transition(list, {kInstances, kGroups, kBatches}, ResourceState::CopyDestination);
    }
    if (step)
    {
        step = impl.Copy(list, impl.sceneStaging, 0, impl.buffers[kInstances], 0, instanceBytes);
    }
    if (step)
    {
        step =
            impl.Copy(list, impl.sceneStaging, instanceBytes, impl.buffers[kGroups], 0, groupBytes);
    }
    if (step)
    {
        step = impl.Copy(list, impl.sceneStaging, instanceBytes + groupBytes,
                         impl.buffers[kBatches], 0, batchBytes);
    }
    if (step)
    {
        step = impl.Transition(list, {kInstances, kGroups, kBatches}, ResourceState::ShaderRead);
    }
    if (!step)
    {
        return step;
    }
    impl.instanceCount = static_cast<u32>(instances.Size());
    impl.batchCount = static_cast<u32>(batches.Size());
    impl.visibleCapacity = scene.VisibleCapacity();
    impl.sceneUploaded = true;
    return Status<Error>::Ok();
}

Status<Error> GpuCullingPipeline::RecordHiZUpload(CommandList& list,
                                                  const HiZPyramid& pyramid) noexcept
{
    if (m_impl == nullptr)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu culling: the pipeline is empty");
    }
    Impl& impl = *m_impl;
    const Span<const f32> texels = pyramid.Texels();
    if (pyramid.Empty() || texels.Size() * sizeof(f32) > impl.hiZBytes ||
        impl.desc.hiZMaxWidth == 0)
    {
        return Fail(ErrorCode::OutOfRange, "gpu culling: the pyramid exceeds the Hi-Z capacity");
    }
    Status<Error> step = impl.device->WriteBuffer(
        impl.hiZStaging, 0, AsBytes(texels.Data(), texels.Size() * sizeof(f32)));
    if (step)
    {
        step = impl.PrepareStaging(list);
    }
    if (step)
    {
        step = impl.Transition(list, {kHiZ}, ResourceState::CopyDestination);
    }
    if (step)
    {
        step =
            impl.Copy(list, impl.hiZStaging, 0, impl.buffers[kHiZ], 0, texels.Size() * sizeof(f32));
    }
    if (step)
    {
        step = impl.Transition(list, {kHiZ}, ResourceState::ShaderRead);
    }
    if (!step)
    {
        return step;
    }
    impl.hiZLevelCount = pyramid.LevelCount();
    for (u32 level = 0; level < kHiZMaxLevels; ++level)
    {
        impl.hiZLevels[level] = pyramid.Level(level);
    }
    impl.hiZReady = true;
    return Status<Error>::Ok();
}

Status<Error> GpuCullingPipeline::RecordHiZBuild(CommandList& list, BufferHandle depth, u32 width,
                                                 u32 height, u32 rowPitch) noexcept
{
    if (m_impl == nullptr)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu culling: the pipeline is empty");
    }
    Impl& impl = *m_impl;
    if (!depth.IsValid() || impl.desc.hiZMaxWidth == 0 || width > impl.desc.hiZMaxWidth ||
        height > impl.desc.hiZMaxHeight)
    {
        return Fail(ErrorCode::OutOfRange,
                    "gpu culling: the depth buffer exceeds the Hi-Z capacity");
    }
    HiZLevel levels[kHiZMaxLevels]{};
    u32 texels = 0;
    const u32 levelCount = ComputeHiZLayout(width, height, levels, &texels);
    if (levelCount == 0)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu culling: the depth buffer has no texels");
    }
    // Find or create the bind group for this depth source.
    BindGroupHandle group{};
    for (u32 index = 0; index < impl.hiZSourceCount; ++index)
    {
        if (impl.hiZSources[index] == depth)
        {
            group = impl.hiZGroups[index];
        }
    }
    if (!group.IsValid())
    {
        if (impl.hiZSourceCount >= kMaxHiZSources)
        {
            return Fail(ErrorCode::OutOfRange, "gpu culling: too many distinct Hi-Z depth sources");
        }
        const BindingResource resources[] = {
            Resource(BindingType::ReadOnlyStorageBuffer, depth),
            Resource(BindingType::StorageBuffer, impl.buffers[kHiZ]),
        };
        const Status<Error> made =
            impl.MakeGroup(impl.hiZGroupLayout, Span<const BindingResource>(resources, 2), group);
        if (!made)
        {
            return made;
        }
        impl.hiZSources[impl.hiZSourceCount] = depth;
        impl.hiZGroups[impl.hiZSourceCount] = group;
        ++impl.hiZSourceCount;
    }

    Status<Error> step = impl.Transition(list, {kHiZ}, ResourceState::ShaderWrite);
    if (step)
    {
        step = list.SetPipeline(impl.hiZPipeline);
    }
    if (step)
    {
        step = list.SetBindGroup(cyber::rhi::BindGroupSlot::PerFrame, group);
    }
    for (u32 level = 0; step && level < levelCount; ++level)
    {
        HiZPassConstants constants{};
        if (level == 0)
        {
            constants.source[0] = 0;
            constants.source[1] = width;
            constants.source[2] = height;
            constants.source[3] = rowPitch == 0 ? width : rowPitch;
        }
        else
        {
            constants.source[0] = levels[level - 1].offset;
            constants.source[1] = levels[level - 1].width;
            constants.source[2] = levels[level - 1].height;
            constants.source[3] = levels[level - 1].width;
            // Level k reads level k - 1, which the previous dispatch wrote.
            step = impl.Transition(list, {kHiZ}, ResourceState::ShaderWrite);
            if (!step)
            {
                break;
            }
        }
        constants.destination[0] = levels[level].offset;
        constants.destination[1] = levels[level].width;
        constants.destination[2] = levels[level].height;
        constants.destination[3] = level == 0 ? 0U : 1U;
        step = list.PushConstants(cyber::rhi::ShaderStageBit(ShaderStage::Compute), 0,
                                  AsBytes(&constants, sizeof(constants)));
        if (step)
        {
            cyber::rhi::DispatchArguments dispatch{};
            dispatch.groupCountX =
                (levels[level].width + kHiZThreadsPerAxis - 1) / kHiZThreadsPerAxis;
            dispatch.groupCountY =
                (levels[level].height + kHiZThreadsPerAxis - 1) / kHiZThreadsPerAxis;
            step = list.Dispatch(dispatch);
        }
    }
    if (step)
    {
        step = impl.Transition(list, {kHiZ}, ResourceState::ShaderRead);
    }
    if (!step)
    {
        return step;
    }
    impl.hiZLevelCount = levelCount;
    for (u32 level = 0; level < kHiZMaxLevels; ++level)
    {
        impl.hiZLevels[level] = level < levelCount ? levels[level] : HiZLevel{};
    }
    impl.hiZReady = true;
    return Status<Error>::Ok();
}

Status<Error> GpuCullingPipeline::RecordCull(CommandList& list, const GpuCullView& view) noexcept
{
    if (m_impl == nullptr)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu culling: the pipeline is empty");
    }
    Impl& impl = *m_impl;
    if (!impl.sceneUploaded)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu culling: no scene has been uploaded");
    }
    const bool occlusion = (view.flags & kGpuCullOcclusion) != 0;
    if (occlusion && !impl.hiZReady)
    {
        return Fail(ErrorCode::InvalidArgument,
                    "gpu culling: occlusion culling needs a Hi-Z upload or build");
    }

    const u32 groups = (impl.instanceCount + kGpuCullThreadsPerGroup - 1) / kGpuCullThreadsPerGroup;
    const u32 groupsX = groups < kMaxGroupsX ? (groups == 0 ? 1 : groups) : kMaxGroupsX;
    const u32 groupsY = groups == 0 ? 1 : (groups + groupsX - 1) / groupsX;

    GpuCullConstants constants{};
    for (u32 plane = 0; plane < 6; ++plane)
    {
        constants.planes[plane][0] = view.frustum.planes[plane].normal.x;
        constants.planes[plane][1] = view.frustum.planes[plane].normal.y;
        constants.planes[plane][2] = view.frustum.planes[plane].normal.z;
        constants.planes[plane][3] = view.frustum.planes[plane].distance;
    }
    for (u32 column = 0; column < 4; ++column)
    {
        const cyber::math::Vec4& value = view.occlusionViewProjection[column];
        constants.occlusionColumns[column][0] = value.x;
        constants.occlusionColumns[column][1] = value.y;
        constants.occlusionColumns[column][2] = value.z;
        constants.occlusionColumns[column][3] = value.w;
    }
    constants.cameraScale[0] = view.cameraPosition.x;
    constants.cameraScale[1] = view.cameraPosition.y;
    constants.cameraScale[2] = view.cameraPosition.z;
    constants.cameraScale[3] = view.projectionScalePixels;
    constants.nearPad[0] = view.nearDistance;
    constants.counts[0] = impl.instanceCount;
    constants.counts[1] = impl.batchCount;
    constants.counts[2] = view.flags & kGpuCullAll;
    constants.counts[3] = groupsX;
    constants.hiZInfo[0] = occlusion ? impl.hiZLevels[0].width : 0;
    constants.hiZInfo[1] = occlusion ? impl.hiZLevels[0].height : 0;
    constants.hiZInfo[2] = occlusion ? impl.hiZLevelCount : 0;
    constants.hiZInfo[3] = impl.visibleCapacity;

    const u64 slotOffset = static_cast<u64>(impl.constantSlot) * kConstantSlotBytes;
    impl.constantSlot = (impl.constantSlot + 1) % kGpuCullConstantSlots;
    Status<Error> step = impl.device->WriteBuffer(impl.constantStaging, slotOffset,
                                                  AsBytes(&constants, sizeof(constants)));
    if (step)
    {
        step = impl.PrepareStaging(list);
    }
    if (step)
    {
        step = impl.Transition(list, {kConstants, kBatchCounts, kCounters},
                               ResourceState::CopyDestination);
    }
    if (step)
    {
        step = impl.Copy(list, impl.constantStaging, slotOffset, impl.buffers[kConstants], 0,
                         sizeof(constants));
    }
    if (step)
    {
        step = impl.Copy(list, impl.zeroSource, 0, impl.buffers[kBatchCounts], 0,
                         static_cast<u64>(impl.batchCount) * sizeof(u32));
    }
    if (step)
    {
        step = impl.Copy(list, impl.zeroSource, 0, impl.buffers[kCounters], 0,
                         sizeof(GpuCullCounters));
    }
    if (step)
    {
        step = impl.Transition(list, {kConstants}, ResourceState::ConstantBuffer);
    }
    if (step)
    {
        step =
            impl.Transition(list, {kBatchCounts, kVisible, kCounters}, ResourceState::ShaderWrite);
    }
    if (step)
    {
        step = impl.Transition(list, {kHiZ}, ResourceState::ShaderRead);
    }
    impl.stats = GpuCullingPipelineStats{};
    if (step && impl.instanceCount > 0)
    {
        step = list.SetPipeline(impl.cullPipeline);
        if (step)
        {
            step = list.SetBindGroup(cyber::rhi::BindGroupSlot::PerFrame, impl.cullGroup);
        }
        if (step)
        {
            cyber::rhi::DispatchArguments dispatch{};
            dispatch.groupCountX = groupsX;
            dispatch.groupCountY = groupsY;
            step = list.Dispatch(dispatch);
            impl.stats.cullGroupsX = groupsX;
            impl.stats.cullGroupsY = groupsY;
            ++impl.stats.cullDispatches;
        }
        if (step)
        {
            // Compaction reads the counts the culling pass's atomics wrote.
            step = impl.Transition(list, {kBatchCounts, kCounters}, ResourceState::ShaderWrite);
        }
    }
    if (step)
    {
        step = impl.Transition(list, {kArguments}, ResourceState::ShaderWrite);
    }
    if (step)
    {
        step = list.SetPipeline(impl.compactPipeline);
    }
    if (step)
    {
        step = list.SetBindGroup(cyber::rhi::BindGroupSlot::PerFrame, impl.compactGroup);
    }
    if (step)
    {
        step = list.Dispatch(cyber::rhi::DispatchArguments{1, 1, 1});
        ++impl.stats.cullDispatches;
    }
    return step;
}

Status<Error> GpuCullingPipeline::RecordAcquireForDraw(CommandList& list) noexcept
{
    if (m_impl == nullptr)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu culling: the pipeline is empty");
    }
    Status<Error> step =
        m_impl->Transition(list, {kArguments, kCounters}, ResourceState::IndirectArgument);
    if (step)
    {
        step = m_impl->Transition(list, {kVisible}, ResourceState::VertexBuffer);
    }
    return step;
}

Status<Error> GpuCullingPipeline::RecordDraw(CommandList& list) noexcept
{
    if (m_impl == nullptr)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu culling: the pipeline is empty");
    }
    Impl& impl = *m_impl;
    if (!impl.sceneUploaded)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu culling: no scene has been uploaded");
    }
    if (impl.states[kArguments] != ResourceState::IndirectArgument ||
        impl.states[kCounters] != ResourceState::IndirectArgument ||
        impl.states[kVisible] != ResourceState::VertexBuffer)
    {
        return Fail(ErrorCode::InvalidArgument,
                    "gpu culling: RecordAcquireForDraw must precede RecordDraw");
    }
    Status<Error> step =
        list.SetVertexBuffer(kGpuDrivenInstanceStreamBinding, impl.buffers[kVisible], 0);
    if (step)
    {
        cyber::rhi::IndirectCountArguments arguments{};
        arguments.argumentBuffer = impl.buffers[kArguments];
        arguments.argumentOffsetBytes = 0;
        arguments.countBuffer = impl.buffers[kCounters];
        arguments.countOffsetBytes = kGpuCullDrawCountOffset;
        arguments.maxDrawCount = impl.batchCount;
        arguments.strideBytes = cyber::rhi::kIndirectDrawIndexedStrideBytes;
        step = list.DrawIndexedIndirectCount(arguments);
        impl.stats.drawCalls = step.IsOk() ? 1 : 0;
    }
    return step;
}

Status<Error> GpuCullingPipeline::RecordReadback(CommandList& list) noexcept
{
    if (m_impl == nullptr)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu culling: the pipeline is empty");
    }
    Impl& impl = *m_impl;
    Status<Error> step = impl.PrepareStaging(list);
    if (step)
    {
        step = impl.Transition(list, {kCounters, kBatchCounts, kVisible, kArguments, kHiZ},
                               ResourceState::CopySource);
    }
    if (step)
    {
        step = impl.Copy(list, impl.buffers[kCounters], 0, impl.counterReadback, 0,
                         sizeof(GpuCullCounters));
    }
    if (step)
    {
        step = impl.Copy(list, impl.buffers[kBatchCounts], 0, impl.batchCountReadback, 0,
                         static_cast<u64>(impl.batchCount) * sizeof(u32));
    }
    if (step)
    {
        step = impl.Copy(list, impl.buffers[kVisible], 0, impl.visibleReadback, 0,
                         static_cast<u64>(impl.visibleCapacity) * sizeof(u32));
    }
    if (step)
    {
        step = impl.Copy(list, impl.buffers[kArguments], 0, impl.argumentReadback, 0,
                         static_cast<u64>(impl.batchCount) *
                             cyber::rhi::kIndirectDrawIndexedStrideBytes);
    }
    if (step && impl.hiZReady)
    {
        HiZLevel levels[kHiZMaxLevels]{};
        u32 texels = 0;
        (void)ComputeHiZLayout(impl.hiZLevels[0].width, impl.hiZLevels[0].height, levels, &texels);
        step = impl.Copy(list, impl.buffers[kHiZ], 0, impl.hiZReadback, 0,
                         static_cast<u64>(texels) * sizeof(f32));
        impl.hiZReadbackRecorded = step.IsOk();
    }
    return step;
}

Status<Error> GpuCullingPipeline::ReadBack(GpuCullOutput& output) noexcept
{
    if (m_impl == nullptr)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu culling: the pipeline is empty");
    }
    Impl& impl = *m_impl;
    cyber::rhi::Device& device = *impl.device;
    if (!output.batchCounts.TryResize(impl.batchCount) ||
        !output.visibleInstances.TryResize(impl.visibleCapacity) ||
        !output.draws.TryResize(impl.batchCount))
    {
        return Fail(ErrorCode::Internal, "gpu culling: readback allocation failed");
    }
    GpuCullCounters counters{};
    Status<Error> step = device.ReadBuffer(
        impl.counterReadback, 0, Span<Byte>(reinterpret_cast<Byte*>(&counters), sizeof(counters)));
    if (step && impl.batchCount > 0)
    {
        step = device.ReadBuffer(impl.batchCountReadback, 0,
                                 Span<Byte>(reinterpret_cast<Byte*>(output.batchCounts.Data()),
                                            output.batchCounts.Size() * sizeof(u32)));
    }
    if (step && impl.visibleCapacity > 0)
    {
        step = device.ReadBuffer(impl.visibleReadback, 0,
                                 Span<Byte>(reinterpret_cast<Byte*>(output.visibleInstances.Data()),
                                            output.visibleInstances.Size() * sizeof(u32)));
    }
    if (step && impl.batchCount > 0)
    {
        static_assert(sizeof(cyber::rhi::DrawIndexedArguments) ==
                          cyber::rhi::kIndirectDrawIndexedStrideBytes,
                      "DrawIndexedArguments is the indirect layout");
        step = device.ReadBuffer(
            impl.argumentReadback, 0,
            Span<Byte>(reinterpret_cast<Byte*>(output.draws.Data()),
                       output.draws.Size() * sizeof(cyber::rhi::DrawIndexedArguments)));
    }
    if (!step)
    {
        return step;
    }
    const u32 drawCount =
        counters.drawCount < impl.batchCount ? counters.drawCount : impl.batchCount;
    (void)output.draws.TryResize(drawCount);
    output.counters = counters;
    return Status<Error>::Ok();
}

Status<Error> GpuCullingPipeline::ReadBackHiZ(Span<f32> texels) noexcept
{
    if (m_impl == nullptr || !m_impl->hiZReadbackRecorded)
    {
        return Fail(ErrorCode::InvalidArgument, "gpu culling: no Hi-Z readback was recorded");
    }
    if (texels.Size() * sizeof(f32) > m_impl->hiZBytes)
    {
        return Fail(ErrorCode::OutOfRange, "gpu culling: the Hi-Z readback is smaller");
    }
    return m_impl->device->ReadBuffer(
        m_impl->hiZReadback, 0,
        Span<Byte>(reinterpret_cast<Byte*>(texels.Data()), texels.Size() * sizeof(f32)));
}

bool GpuCullingPipeline::IsValid() const noexcept
{
    return m_impl != nullptr;
}

const GpuCullingPipelineStats& GpuCullingPipeline::Stats() const noexcept
{
    static const GpuCullingPipelineStats kEmpty{};
    return m_impl != nullptr ? m_impl->stats : kEmpty;
}

BufferHandle GpuCullingPipeline::ArgumentBuffer() const noexcept
{
    return m_impl != nullptr ? m_impl->buffers[kArguments] : BufferHandle{};
}

BufferHandle GpuCullingPipeline::CounterBuffer() const noexcept
{
    return m_impl != nullptr ? m_impl->buffers[kCounters] : BufferHandle{};
}

BufferHandle GpuCullingPipeline::VisibleBuffer() const noexcept
{
    return m_impl != nullptr ? m_impl->buffers[kVisible] : BufferHandle{};
}

u32 GpuCullingPipeline::BatchCount() const noexcept
{
    return m_impl != nullptr ? m_impl->batchCount : 0;
}

}  // namespace cyber::renderer
