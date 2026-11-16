// The Vulkan backend's HLSL compiler (src/vulkan/hlsl_spirv.h), without a GPU:
// every HLSL pattern the engine's shaders use compiles, the module is structurally
// sound (header, instruction lengths, id bound, one definition per id, every used id
// defined), bindings resolve through a layout like the device's, vertex outputs link
// to pixel inputs by semantic, and malformed or unsupported input fails with a
// located diagnostic instead of a crash. The GPU half -- the same HLSL producing the
// same results on D3D12 and Vulkan -- is in rhi_backend_gpu_parity and the renderer
// tests run on both backends.

#include "../src/vulkan/hlsl_spirv.h"

#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/test/test.h>

#include <cstdio>
#include <cstring>

using cyber::core::u32;
using cyber::core::usize;
using cyber::memory::HeapAllocator;
using namespace cyber::rhi;
using namespace cyber::rhi::detail;

namespace
{

// Opcodes with a result id at word 2 (after the result type at word 1), or at
// word 1 for type declarations and labels. Enough to check definitions.
bool HasResultTypeAndId(u32 op) noexcept
{
    switch (op)
    {
    case 1:   // OpUndef
    case 12:  // OpExtInst
    case 41:  // OpConstantTrue
    case 42:
    case 43:
    case 44:
    case 46:
    case 54:  // OpFunction
    case 55:  // OpFunctionParameter
    case 57:  // OpFunctionCall
    case 59:  // OpVariable
        return true;
    default:
        break;
    }
    if (op == 99 || op == 228)  // OpImageWrite, OpAtomicStore: no result
    {
        return false;
    }
    return (op >= 61 && op <= 68 && op != 62 && op != 63 && op != 64) || (op >= 77 && op <= 84) ||
           (op >= 86 && op <= 107) || (op >= 109 && op <= 209) || (op >= 227 && op <= 242);
}

bool HasResultIdOnly(u32 op) noexcept
{
    return op == 11 || (op >= 19 && op <= 33) || op == 248;  // import, types, label
}

// Structural checks a validator would also make. Returns a description of the
// first problem, or null.
const char* CheckModule(const cyber::memory::DynamicArray<u32>& words) noexcept
{
    if (words.Size() < 5 || words[0] != 0x07230203U || words[1] != 0x00010500U)
    {
        return "bad header";
    }
    const u32 bound = words[3];
    if (bound == 0 || bound > 1000000)
    {
        return "bad bound";
    }
    static bool defined[1000000];
    std::memset(defined, 0, bound);
    usize at = 5;
    bool sawFunction = false;
    bool sawEntry = false;
    while (at < words.Size())
    {
        const u32 count = words[at] >> 16;
        const u32 op = words[at] & 0xFFFF;
        if (count == 0 || at + count > words.Size())
        {
            return "bad instruction length";
        }
        u32 result = 0;
        if (HasResultTypeAndId(op) && count >= 3)
        {
            result = words[at + 2];
            if (words[at + 1] == 0 || words[at + 1] >= bound || !defined[words[at + 1]])
            {
                return "result type used before it is defined";
            }
        }
        else if (HasResultIdOnly(op) && count >= 2)
        {
            result = words[at + 1];
        }
        if (result != 0)
        {
            if (result >= bound)
            {
                return "id beyond the bound";
            }
            if (defined[result])
            {
                return "id defined twice";
            }
            defined[result] = true;
        }
        sawFunction = sawFunction || op == 54;
        sawEntry = sawEntry || op == 15;
        at += count;
    }
    if (!sawFunction || !sawEntry)
    {
        return "missing function or entry point";
    }
    return nullptr;
}

bool CompileOk(HeapAllocator& heap, ShaderStage stage, const char* source, const char* name,
               hlsl::CompileOutput* out, const hlsl::CompileDesc* base = nullptr) noexcept
{
    hlsl::CompileDesc desc = base != nullptr ? *base : hlsl::CompileDesc{};
    desc.source = source;
    desc.sourceBytes = std::strlen(source);
    desc.stage = stage;
    desc.sourceName = name;
    desc.features.drawParameters = true;
    desc.features.storageImageReadWithoutFormat = true;
    desc.features.storageImageWriteWithoutFormat = true;
    const auto compiled = hlsl::Compile(desc, heap, out);
    if (!compiled)
    {
        std::printf("  %s: %s\n", name, compiled.Error().message);
        return false;
    }
    const char* problem = CheckModule(out->words);
    if (problem != nullptr)
    {
        std::printf("  %s: %s\n", name, problem);
        return false;
    }
    return true;
}

// The engine's shader patterns, condensed from the renderer and samples.
constexpr const char* kCullCompute = R"HLSL(
cbuffer CullConstants : register(b0, space0)
{
    float4 Planes[6];
    float4 OcclusionColumns[4];
    float4 CameraScale;
    float4 NearPad;
    uint4 Counts;
    uint4 HiZInfo;
};
struct DrawBatch { uint indexCount; uint firstIndex; int vertexOffset; uint visibleBase; uint capacity; uint3 reserved; };
struct Instance { float3 boundsMin; uint drawGroup; float3 boundsMax; uint instanceId; };
StructuredBuffer<Instance> Instances : register(t0, space0);
StructuredBuffer<DrawBatch> Batches : register(t2, space0);
StructuredBuffer<float> HiZ : register(t3, space0);
RWStructuredBuffer<uint> BatchCounts : register(u0, space0);
RWStructuredBuffer<uint> Visible : register(u1, space0);
static const float kFloatMax = 3.402823466e+38;
static const uint kHiZMaxLevels = 16;
groupshared uint g_visible;
uint3 HiZShape(uint level)
{
    uint width = HiZInfo.x; uint height = HiZInfo.y; uint offset = 0;
    for (uint l = 0; l < level; ++l) { offset += width * height; width = (width + 1) / 2; height = (height + 1) / 2; }
    return uint3(width, height, offset);
}
bool Occluded(float3 boundsMin, float3 boundsMax)
{
    precise float minX = kFloatMax;
    precise float maxZ = -kFloatMax;
    [unroll]
    for (uint corner = 0; corner < 8; ++corner)
    {
        float px = (corner & 1) != 0 ? boundsMax.x : boundsMin.x;
        precise float4 clip = (OcclusionColumns[0] * px + OcclusionColumns[1] * px) + OcclusionColumns[3];
        if (clip.w > 1.0e-5) { precise float x = clip.x / clip.w; minX = x < minX ? x : minX; maxZ = max(maxZ, clip.z); }
    }
    uint lx0 = (uint)max(minX, 0.0);
    [loop]
    for (uint step = 0; step + 1 < kHiZMaxLevels; ++step)
    {
        if (step + 1 >= HiZInfo.z || lx0 <= 1) break;
        lx0 >>= 1;
    }
    uint3 shape = HiZShape(lx0);
    return maxZ < HiZ[shape.z + min(lx0, shape.x - 1)];
}
[numthreads(64, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    if (groupIndex == 0) g_visible = 0;
    GroupMemoryBarrierWithGroupSync();
    uint index = (groupId.y * Counts.w + groupId.x) * 64 + groupIndex;
    if (index < Counts.x)
    {
        Instance instance = Instances[index];
        float thresholds[4] = {1.0, 2.0, 3.0, 4.0};
        bool visible = !Occluded(instance.boundsMin, instance.boundsMax) && thresholds[index & 3] > 0.0;
        for (uint p = 0; p < 6; ++p)
        {
            float4 plane = Planes[p];
            if (dot(plane.xyz, instance.boundsMax) + plane.w < 0.0) { visible = false; break; }
        }
        if (visible)
        {
            uint slot;
            InterlockedAdd(BatchCounts[instance.drawGroup], 1, slot);
            DrawBatch batch = Batches[instance.drawGroup];
            if (slot < batch.capacity) Visible[batch.visibleBase + slot] = instance.instanceId;
            InterlockedAdd(g_visible, 1);
        }
    }
}
)HLSL";

constexpr const char* kScanCompute = R"HLSL(
RWStructuredBuffer<uint> Data : register(u0, space0);
groupshared uint g_sums[256];
[numthreads(256, 1, 1)]
void main(uint thread : SV_GroupIndex)
{
    g_sums[thread] = Data[thread];
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint offset = 1; offset < 256; offset <<= 1)
    {
        uint add = thread >= offset ? g_sums[thread - offset] : 0;
        GroupMemoryBarrierWithGroupSync();
        g_sums[thread] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    Data[thread] = g_sums[thread] - asuint(asint(1)) + 1u;
}
)HLSL";

constexpr const char* kVertexStruct = R"HLSL(
cbuffer Frame : register(b0, space0) { float4 ViewProjection[4]; float4 SunDirection; float4 CameraPosition; };
StructuredBuffer<float4> Instances : register(t0, space0);
struct Input { float3 position : TEXCOORD0; float3 normal : TEXCOORD1; uint instanceId : TEXCOORD2; };
struct Output { float4 position : SV_Position; float3 normal : NORMAL; float3 color : COLOR; float3 world : WORLDPOS; };
Output main(Input input)
{
    float4 placement = Instances[input.instanceId * 2];
    float3 world = placement.xyz + input.position * placement.w;
    Output output;
    output.position = ViewProjection[0] * world.x + ViewProjection[1] * world.y + ViewProjection[2] * world.z + ViewProjection[3];
    output.normal = input.normal;
    output.color = Instances[input.instanceId * 2 + 1].rgb;
    output.world = world;
    return output;
}
)HLSL";

// Inputs in a different order from the vertex outputs, and a subset of them.
constexpr const char* kPixelStruct = R"HLSL(
cbuffer Frame : register(b0, space0) { float4 ViewProjection[4]; float4 SunDirection; float4 CameraPosition; };
struct Input { float4 position : SV_Position; float3 world : WORLDPOS; float3 normal : NORMAL; };
float4 main(Input input) : SV_Target
{
    float3 normal = normalize(input.normal);
    float diffuse = saturate(dot(normal, -SunDirection.xyz));
    float fog = 1.0 - exp(-length(input.world - CameraPosition.xyz) * 0.0014);
    float3 color = lerp(diffuse.xxx, float3(0.60, 0.70, 0.84), fog);
    return float4(pow(saturate(color), 1.0 / 2.2), 1.0);
}
)HLSL";

constexpr const char* kMatrixVertex = R"HLSL(
cbuffer ObjectTransform : register(b0, space3) { float4x4 objectToClip; };
cbuffer DrawConstants : register(b1, space3) { float4x4 m : packoffset(c0); float4 baseColor : packoffset(c4); };
struct Input { float2 position : TEXCOORD0; };
float4 main(Input input) : SV_Position
{
    float4 p = mul(objectToClip, float4(input.position, 0.0, 1.0));
    float4 q = mul(float4(input.position, 0.0, 1.0), transpose(m));
    float3x3 r = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    return p + q * baseColor.x + float4(mul(r, p.xyz), 0.0) * 0.0 + float4(m[1].xyz, m._m33) * 0.0;
}
)HLSL";

constexpr const char* kBindlessVertex = R"HLSL(
cbuffer DrawIndex : register(b0, space0) { uint objectIndex; };
StructuredBuffer<float4x4> objects[10000] : register(t10000, space2);
struct Input { float2 position : TEXCOORD0; };
float4 main(Input input, uint vertexId : SV_VertexID, uint instance : SV_InstanceID) : SV_Position
{
    return mul(objects[objectIndex][0], float4(input.position, (float)vertexId * 0.0, 1.0 + instance * 0.0));
}
)HLSL";

constexpr const char* kTextures = R"HLSL(
Texture2D albedo : register(t0, space0);
Texture2D<float> shadowDepth : register(t0, space1);
Texture2D<uint> pageTable : register(t1, space0);
Texture2DArray<float4> layers : register(t2, space0);
TextureCube<float4> sky : register(t3, space0);
Texture2D<float4> Tex[6] : register(t4, space0);
SamplerState albedoSampler : register(s0, space0);
SamplerComparisonState shadowSampler : register(s0, space1);
float4 main(float4 position : SV_Position, float2 uv : TEXCOORD0, nointerpolation uint index : TEXCOORD1) : SV_Target
{
    float visibility = shadowDepth.SampleCmpLevelZero(shadowSampler, position.xy / 64.0, 0.6);
    uint entry = pageTable.Load(int3(int2(uv), 0));
    float4 c = albedo.Sample(albedoSampler, uv) + layers.SampleLevel(albedoSampler, float3(uv, 1), 0);
    c += sky.SampleLevel(albedoSampler, float3(uv, 1.0), 2.0) + Tex[NonUniformResourceIndex(index % 6)].Load(int3(0, 0, 0));
    uint w, h, levels;
    albedo.GetDimensions(0, w, h, levels);
    if (c.a < 0.01) discard;
    clip(c.r - 0.001);
    return c * visibility + float((entry >> 20u) & 0x1Fu) + float(w + h + levels) * 0.0 + ddx(uv.x);
}
)HLSL";

constexpr const char* kVirtualTexture = R"HLSL(
Texture2D<uint> g_vtPageTable : register(t0, space0);
Texture2D<float4> g_vtAtlas : register(t1, space0);
RWStructuredBuffer<uint> g_vtFeedback : register(u1, space0);
struct VtParams { uint width; uint height; uint pageSize; uint border; uint stride; uint mipCount; uint reserved0; uint reserved1; };
float4 VtSample(VtParams p, float2 uv, float lod)
{
    float level = isfinite(lod) ? max(lod, 0.0f) : 0.0f;
    uint mip = min((uint)level, p.mipCount - 1u);
    float2 f = uv - floor(uv);
    uint2 pages = uint2(max((p.width >> mip) / p.pageSize, 1u), max((p.height >> mip) / p.pageSize, 1u));
    uint2 page = min((uint2)(f * (float2)pages), pages - 1u);
    uint entry = g_vtPageTable.Load(int3((int2)page, (int)mip));
    uint2 cache = uint2(entry & 0x3FFu, (entry >> 10u) & 0x3FFu);
    float2 a = (float2)(cache * p.stride) + (float)p.border - 0.5f;
    float2 base = floor(a);
    float2 w = a - base;
    int2 x0 = (int2)base;
    float4 t00 = g_vtAtlas.Load(int3(x0, 0));
    float4 t10 = g_vtAtlas.Load(int3(x0 + int2(1, 0), 0));
    return lerp(t00, t10, w.x);
}
void VtFeedback(VtParams p, float2 uv, float lod)
{
    uint index = 0u;
    for (uint m = 0u; m < (uint)lod; ++m) { index += max((p.width >> m) / p.pageSize, 1u); }
    InterlockedAdd(g_vtFeedback[index], 1u);
}
StructuredBuffer<float4> Input : register(t2, space0);
StructuredBuffer<uint> Params : register(t3, space0);
RWStructuredBuffer<float4> Output : register(u0, space0);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    VtParams p;
    p.width = Params[0]; p.height = Params[1]; p.pageSize = Params[2]; p.border = Params[3];
    p.stride = Params[4]; p.mipCount = Params[5]; p.reserved0 = 0; p.reserved1 = 0;
    float4 q = Input[id.x];
    Output[id.x] = VtSample(p, q.xy, q.z);
    VtFeedback(p, q.xy, q.z);
}
)HLSL";

constexpr const char* kStorageImage = R"HLSL(
RWTexture2D<float4> Target : register(u0, space0);
RWByteAddressBuffer Raw : register(u1, space0);
ByteAddressBuffer RawIn : register(t0, space0);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 local : SV_GroupThreadID)
{
    uint3 v = RawIn.Load3(id.x * 12);
    Target[id.xy] = Target[id.xy] * 0.5 + float4(v, 1.0);
    uint original;
    Raw.InterlockedAdd(0, 1, original);
    Raw.Store2(8, uint2(original, local.x));
    int i = -5;
    do { i += 2; } while (i < 0);
    while (i > 0) { i--; if (i == 3) continue; }
    Raw.Store(16, uint(i) | (countbits(original) << 8) | firstbithigh(original));
}
)HLSL";

constexpr const char* kCyberGenerated = R"HLSL(
struct Input
{
    float3 position : TEXCOORD0;
    float2 uv : TEXCOORD1;
};
struct Output
{
    float4 clipPosition : SV_Position;
    float2 texcoord : TEXCOORD0;
};
cbuffer Cube_Uniforms0 : register(b0)
{
    float4x4 modelViewProjection : packoffset(c0);
};
Output main(Input input)
{
    Output output;
    output.clipPosition = mul(modelViewProjection, float4(input.position, 1.0));
    output.texcoord = input.uv;
    return output;
}
)HLSL";

// A layout like the device builds: set = space, explicit bindings.
class TestResolver final : public hlsl::BindingResolver
{
public:
    bool Resolve(hlsl::RegisterClass registerClass, u32 registerIndex, u32 space,
                 hlsl::ResolvedBinding* out) const noexcept override
    {
        if (registerClass == hlsl::RegisterClass::ConstantBuffer && registerIndex == 0 &&
            space == 0)
        {
            out->kind = hlsl::BindingKind::PushConstant;
            return true;
        }
        if (registerClass == hlsl::RegisterClass::ShaderResource && space == 0 && registerIndex < 3)
        {
            // One storage-buffer array binding covering t0..t2.
            out->kind = hlsl::BindingKind::StorageBuffer;
            out->set = 0;
            out->binding = 7;
            out->arrayCount = 3;
            out->firstElement = registerIndex;
            return true;
        }
        if (registerClass == hlsl::RegisterClass::UnorderedAccess && space == 0 &&
            registerIndex == 0)
        {
            out->kind = hlsl::BindingKind::StorageBuffer;
            out->set = 0;
            out->binding = 9;
            return true;
        }
        return false;
    }
};

}  // namespace

CG_TEST(hlsl_spirv_compiles_engine_shader_patterns)
{
    HeapAllocator heap;
    struct Case
    {
        ShaderStage stage;
        const char* source;
        const char* name;
    };
    const Case cases[] = {
        {ShaderStage::Compute, kCullCompute, "cull"},
        {ShaderStage::Compute, kScanCompute, "scan"},
        {ShaderStage::Vertex, kVertexStruct, "vertex struct"},
        {ShaderStage::Pixel, kPixelStruct, "pixel struct"},
        {ShaderStage::Vertex, kMatrixVertex, "matrices"},
        {ShaderStage::Vertex, kBindlessVertex, "bindless"},
        {ShaderStage::Pixel, kTextures, "textures"},
        {ShaderStage::Compute, kVirtualTexture, "virtual texture"},
        {ShaderStage::Compute, kStorageImage, "storage image"},
        {ShaderStage::Vertex, kCyberGenerated, "cyber generated"},
    };
    for (const Case& c : cases)
    {
        hlsl::CompileOutput out(heap);
        CG_EXPECT_TRUE(CompileOk(heap, c.stage, c.source, c.name, &out));
    }
}

CG_TEST(hlsl_spirv_links_varyings_by_semantic)
{
    HeapAllocator heap;
    hlsl::CompileOutput vertex(heap);
    CG_REQUIRE(CompileOk(heap, ShaderStage::Vertex, kVertexStruct, "vs", &vertex));
    CG_EXPECT_EQ(vertex.outputCount, 3U);
    CG_EXPECT_STREQ(vertex.outputs[0].semantic, "NORMAL");
    CG_EXPECT_STREQ(vertex.outputs[2].semantic, "WORLDPOS");
    hlsl::CompileDesc linked{};
    linked.linkedInputs = vertex.outputs;
    linked.linkedInputCount = vertex.outputCount;
    hlsl::CompileOutput pixel(heap);
    CG_EXPECT_TRUE(CompileOk(heap, ShaderStage::Pixel, kPixelStruct, "ps", &pixel, &linked));

    // An input the vertex stage does not write fails to link.
    constexpr const char* kUnlinked = "float4 main(float4 p : SV_Position, float2 uv : TEXCOORD5) "
                                      ": SV_Target { return uv.xyxy; }";
    hlsl::CompileDesc desc = linked;
    desc.source = kUnlinked;
    desc.sourceBytes = std::strlen(kUnlinked);
    desc.stage = ShaderStage::Pixel;
    hlsl::CompileOutput failed(heap);
    const auto result = hlsl::Compile(desc, heap, &failed);
    CG_EXPECT_FALSE(result.IsOk());
}

CG_TEST(hlsl_spirv_resolves_bindings_through_the_layout)
{
    HeapAllocator heap;
    TestResolver resolver;
    hlsl::CompileDesc desc{};
    desc.resolver = &resolver;
    constexpr const char* kShader = R"(
cbuffer PC : register(b0, space0) { uint count; uint scale; uint bias; };
StructuredBuffer<uint> a : register(t0, space0);
StructuredBuffer<uint> c : register(t2, space0);
RWStructuredBuffer<uint> o : register(u0, space0);
[numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID) { if (id.x < count) o[id.x] = a[id.x] * scale + c[id.x] + bias; }
)";
    hlsl::CompileOutput out(heap);
    CG_REQUIRE(CompileOk(heap, ShaderStage::Compute, kShader, "resolved", &out, &desc));
    // A register the layout lacks names itself.
    constexpr const char* kMissing = "RWStructuredBuffer<uint> o : register(u3, space1);\n"
                                     "[numthreads(1,1,1)] void main() { o[0] = 1; }";
    desc.source = kMissing;
    desc.sourceBytes = std::strlen(kMissing);
    desc.stage = ShaderStage::Compute;
    hlsl::CompileOutput failed(heap);
    const auto result = hlsl::Compile(desc, heap, &failed);
    CG_REQUIRE(!(result.IsOk()));
    CG_EXPECT_TRUE(std::strstr(result.Error().message, "u3, space1") != nullptr);
    CG_EXPECT_TRUE(std::strstr(result.Error().message, ":2:") != nullptr);
}

CG_TEST(hlsl_spirv_reports_located_diagnostics)
{
    HeapAllocator heap;
    struct Bad
    {
        ShaderStage stage;
        const char* source;
        const char* expect;
    };
    const Bad cases[] = {
        {ShaderStage::Pixel, "float4 main() : SV_Target\n{\n    return float4(1.0, 0.0, 0.0;\n}\n",
         ":3:"},
        {ShaderStage::Compute, "[numthreads(1,1,1)] void main() { undeclared = 1; }", "undeclared"},
        {ShaderStage::Compute, "void main() {}", "numthreads"},
        {ShaderStage::Compute, "[numthreads(1,1,1)] void main() { switch (1) { } }",
         "not supported"},
        {ShaderStage::Vertex, "#define X 1\nfloat4 main() : SV_Position { return 0; }",
         "not supported"},
        {ShaderStage::Compute,
         "int f(int x) { return f(x); } [numthreads(1,1,1)] void main() { f(1); }", "recursion"},
        {ShaderStage::Pixel, "float4 main() : SV_Target { return float2(1, 2); }", "wider"},
        {ShaderStage::Vertex, "float4 main(float3 p : POSITION) : SV_Position { return p.xyzz; }",
         "TEXCOORD"},
    };
    for (const Bad& bad : cases)
    {
        hlsl::CompileDesc desc{};
        desc.source = bad.source;
        desc.sourceBytes = std::strlen(bad.source);
        desc.stage = bad.stage;
        desc.sourceName = "bad.hlsl";
        hlsl::CompileOutput out(heap);
        const auto result = hlsl::Compile(desc, heap, &out);
        CG_EXPECT_FALSE(result.IsOk());
        if (!result.IsOk())
        {
            const bool found = std::strstr(result.Error().message, bad.expect) != nullptr;
            if (!found)
            {
                std::printf("  expected '%s' in: %s\n", bad.expect, result.Error().message);
            }
            CG_EXPECT_TRUE(found);
            CG_EXPECT_TRUE(std::strncmp(result.Error().message, "bad.hlsl:", 9) == 0);
        }
        CG_EXPECT_EQ(out.words.Size(), static_cast<cyber::core::usize>(0));
    }
}

CG_TEST(hlsl_spirv_survives_hostile_input)
{
    HeapAllocator heap;
    // Every prefix of a real shader, deep nesting, and pseudo-random bytes: each
    // must return (success or a diagnostic) without crashing or leaking.
    const usize length = std::strlen(kCullCompute);
    for (usize cut = 0; cut < length; cut += 7)
    {
        hlsl::CompileDesc desc{};
        desc.source = kCullCompute;
        desc.sourceBytes = cut;
        desc.stage = ShaderStage::Compute;
        hlsl::CompileOutput out(heap);
        (void)hlsl::Compile(desc, heap, &out);
    }
    static char nested[20000];
    usize used = 0;
    const char* head = "[numthreads(1,1,1)] void main() { int x = ";
    std::memcpy(nested, head, std::strlen(head));
    used = std::strlen(head);
    for (int index = 0; index < 4000; ++index)
    {
        nested[used++] = '(';
    }
    nested[used++] = '1';
    for (int index = 0; index < 4000; ++index)
    {
        nested[used++] = ')';
    }
    std::memcpy(nested + used, "; }", 3);
    used += 3;
    {
        hlsl::CompileDesc desc{};
        desc.source = nested;
        desc.sourceBytes = used;
        desc.stage = ShaderStage::Compute;
        hlsl::CompileOutput out(heap);
        (void)hlsl::Compile(desc, heap, &out);
    }
    u32 state = 12345;
    char noise[512];
    for (int round = 0; round < 200; ++round)
    {
        for (char& c : noise)
        {
            state = state * 1664525U + 1013904223U;
            const char alphabet[] = "abcxyz019(){}[];,.:?+-*/%&|^~!<>= \n#_uintfloat4";
            c = alphabet[(state >> 24) % (sizeof(alphabet) - 1)];
        }
        hlsl::CompileDesc desc{};
        desc.source = noise;
        desc.sourceBytes = sizeof(noise);
        desc.stage = static_cast<ShaderStage>(round % 3);
        hlsl::CompileOutput out(heap);
        (void)hlsl::Compile(desc, heap, &out);
    }
    CG_EXPECT_TRUE(true);
}
