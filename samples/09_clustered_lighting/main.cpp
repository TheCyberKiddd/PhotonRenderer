#include <cyber/core/error.h>
#include <cyber/core/types.h>
#include <cyber/math/constants.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/renderer/clustered_lighting.h>
#include <cyber/rhi/backend.h>
#include <cyber/rhi/binding.h>
#include <cyber/rhi/buffer.h>
#include <cyber/rhi/capabilities.h>
#include <cyber/rhi/command_list.h>
#include <cyber/rhi/d3d12.h>
#include <cyber/rhi/device.h>
#include <cyber/rhi/pipeline.h>
#include <cyber/rhi/resource_state.h>
#include <cyber/rhi/shader.h>
#include <cyber/rhi/submission.h>
#include <cyber/software/pbr.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

namespace
{

using cyber::core::Error;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::usize;
using cyber::memory::Byte;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using cyber::renderer::ClusterGridDesc;
using cyber::renderer::ClusterLightBound;
using cyber::renderer::ClusterLightLists;
using cyber::rhi::BindGroupHandle;
using cyber::rhi::BindGroupLayoutHandle;
using cyber::rhi::BufferHandle;
using cyber::rhi::CommandList;
using cyber::rhi::Device;
using cyber::rhi::PipelineHandle;
using cyber::rhi::PipelineLayoutHandle;
using cyber::rhi::ShaderHandle;
using cyber::software::PbrLightingData;
using cyber::software::PbrPointLight;
using cyber::software::Varyings;

constexpr u32 kWidth = 320;
constexpr u32 kHeight = 180;
constexpr u32 kTileSize = 16;
constexpr u32 kDepthSlices = 24;
constexpr u32 kLightCount = 1000;
constexpr u32 kMaxLightsPerCluster = 128;
constexpr u64 kUniformBufferBytes = 256;
constexpr float kNearPlane = 0.1f;
constexpr float kFarPlane = 64.0f;
constexpr float kVerticalFov = cyber::math::kPi / 3.0f;
constexpr u32 kThreadsPerGroup = 64;

constexpr char kBuildShader[] = R"HLSL(
cbuffer BuildParams : register(b0, space0)
{
    uint4 Grid0;
    uint4 Grid1;
    float4 Projection;
};

struct Light
{
    float4 positionRange;
    float4 intensity;
};

StructuredBuffer<Light> Lights : register(t0, space0);
RWStructuredBuffer<uint2> Headers : register(u0, space0);
RWStructuredBuffer<uint> Indices : register(u1, space0);

uint DepthSlice(float depth)
{
    float ratio = log(depth / Projection.y) / log(Projection.z / Projection.y);
    return min((uint)floor(ratio * Grid0.w), Grid0.w - 1);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID)
{
    uint clusterCount = Grid1.x * Grid1.y * Grid0.w;
    uint cluster = dispatchId.x;
    if (cluster >= clusterCount)
        return;

    uint tilesX = Grid1.x;
    uint tilesY = Grid1.y;
    uint tileX = cluster % tilesX;
    uint tileY = (cluster / tilesX) % tilesY;
    uint slice = cluster / (tilesX * tilesY);
    uint count = 0;
    uint overflow = 0;

    for (uint lightIndex = 0; lightIndex < Grid1.z; ++lightIndex)
    {
        float4 sphere = Lights[lightIndex].positionRange;
        float centerDepth = -sphere.z;
        float depthMin = centerDepth - sphere.w;
        float depthMax = centerDepth + sphere.w;
        if (depthMax < Projection.y || depthMin > Projection.z)
            continue;

        uint minSlice = DepthSlice(max(depthMin, Projection.y));
        uint maxSlice = DepthSlice(min(depthMax, Projection.z));
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
                    float pixelX = (0.5 + 0.5 * x / (depth * Projection.x * Projection.w)) * Grid0.x;
                    pixelMin.x = min(pixelMin.x, pixelX);
                    pixelMax.x = max(pixelMax.x, pixelX);
                    float y = sphere.y + (axisCorner == 0 ? -sphere.w : sphere.w);
                    float pixelY = (0.5 - 0.5 * y / (depth * Projection.x)) * Grid0.y;
                    pixelMin.y = min(pixelMin.y, pixelY);
                    pixelMax.y = max(pixelMax.y, pixelY);
                }
            }
        }

        if (pixelMax.x < 0.0 || pixelMin.x > Grid0.x ||
            pixelMax.y < 0.0 || pixelMin.y > Grid0.y)
            continue;

        uint minTileX = (uint)clamp(floor((pixelMin.x - 1.0) / Grid0.z), 0.0, (float)(tilesX - 1));
        uint maxTileX = (uint)clamp(floor((pixelMax.x + 1.0) / Grid0.z), 0.0, (float)(tilesX - 1));
        uint minTileY = (uint)clamp(floor((pixelMin.y - 1.0) / Grid0.z), 0.0, (float)(tilesY - 1));
        uint maxTileY = (uint)clamp(floor((pixelMax.y + 1.0) / Grid0.z), 0.0, (float)(tilesY - 1));
        if (slice < minSlice || slice > maxSlice ||
            tileX < minTileX || tileX > maxTileX || tileY < minTileY || tileY > maxTileY)
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

constexpr char kShadeShader[] = R"HLSL(
cbuffer ShadeParams : register(b0, space0)
{
    uint4 Grid0;
    uint4 Grid1;
    float4 DepthEye;
    float4 EyeAmbient;
};

struct Light
{
    float4 positionRange;
    float4 intensity;
};

struct Surface
{
    float4 positionDepth;
    float4 normal;
    float4 baseColorMetallic;
    float4 roughnessAndAlpha;
};

StructuredBuffer<Light> Lights : register(t0, space0);
StructuredBuffer<uint2> Headers : register(t1, space0);
StructuredBuffer<uint> Indices : register(t2, space0);
StructuredBuffer<Surface> Surfaces : register(t3, space0);
RWStructuredBuffer<float4> Output : register(u0, space0);

float Saturate(float value)
{
    return clamp(value, 0.0, 1.0);
}

float3 SafeNormalize(float3 value)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 1.0e-12 ? value * rsqrt(lengthSquared) : float3(0.0, 0.0, 1.0);
}

float SchlickGeometry(float nDotX, float alpha)
{
    float alphaSquared = alpha * alpha;
    float denominator = nDotX + sqrt(alphaSquared + (1.0 - alphaSquared) * nDotX * nDotX);
    return denominator > 0.0 ? 2.0 * nDotX / denominator : 0.0;
}

float3 EvaluateLight(Surface surface, Light light)
{
    float3 offset = light.positionRange.xyz - surface.positionDepth.xyz;
    float distanceSquared = dot(offset, offset);
    if (distanceSquared <= 1.0e-8)
        return 0.0;

    float distance = sqrt(distanceSquared);
    float range = light.positionRange.w;
    float ratio = distance / range;
    float fade = distance < range ? (1.0 - ratio * ratio) * (1.0 - ratio * ratio) : 0.0;
    if (fade <= 0.0)
        return 0.0;

    float3 n = SafeNormalize(surface.normal.xyz);
    float3 v = SafeNormalize(float3(DepthEye.z, DepthEye.w, EyeAmbient.x) - surface.positionDepth.xyz);
    float3 l = offset / distance;
    float nDotV = Saturate(dot(n, v));
    float nDotL = Saturate(dot(n, l));
    if (nDotV <= 0.0 || nDotL <= 0.0)
        return 0.0;

    float metallic = Saturate(surface.baseColorMetallic.w);
    float roughness = clamp(surface.roughnessAndAlpha.x, 0.045, 1.0);
    float alpha = roughness * roughness;
    float3 baseColor = clamp(surface.baseColorMetallic.xyz, 0.0, 1.0);
    float3 halfVector = SafeNormalize(v + l);
    float nDotH = Saturate(dot(n, halfVector));
    float vDotH = Saturate(dot(v, halfVector));
    float alphaSquared = alpha * alpha;
    float distributionDenominator = nDotH * nDotH * (alphaSquared - 1.0) + 1.0;
    float distribution = alphaSquared /
        max(3.14159265358979323846 * distributionDenominator * distributionDenominator, 1.0e-7);
    float geometry = SchlickGeometry(nDotV, alpha) * SchlickGeometry(nDotL, alpha);
    float fresnelFactor = pow(1.0 - vDotH, 5.0);
    float3 f0 = lerp(0.04, baseColor, metallic);
    float3 fresnel = f0 + (1.0 - f0) * fresnelFactor;
    float3 specular = fresnel *
        (distribution * geometry / max(4.0 * nDotV * nDotL, 1.0e-6));
    float3 diffuseWeight = (1.0 - fresnel) * (1.0 - metallic);
    float3 diffuse = diffuseWeight * baseColor / 3.14159265358979323846;
    float3 radiance = max(light.intensity.xyz, 0.0) *
        (fade / max(distanceSquared, 1.0e-4));
    return (diffuse + specular) * radiance * nDotL;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID)
{
    uint pixelIndex = dispatchId.x;
    uint pixelCount = Grid0.x * Grid0.y;
    if (pixelIndex >= pixelCount)
        return;

    Surface surface = Surfaces[pixelIndex];
    uint pixelX = pixelIndex % Grid0.x;
    uint pixelY = pixelIndex / Grid0.x;
    float depth = surface.positionDepth.w;
    uint clusterCount = Grid1.x * Grid1.y * Grid0.w;
    bool validCluster = depth >= DepthEye.x && depth <= DepthEye.y;
    uint cluster = 0;
    if (validCluster)
    {
        uint tileX = pixelX / Grid0.z;
        uint tileY = pixelY / Grid0.z;
        float ratio = log(depth / DepthEye.x) / log(DepthEye.y / DepthEye.x);
        uint slice = min((uint)floor(ratio * Grid0.w), Grid0.w - 1);
        cluster = (slice * Grid1.y + tileY) * Grid1.x + tileX;
        validCluster = cluster < clusterCount;
    }

    float3 color = max(surface.baseColorMetallic.xyz, 0.0) *
        (1.0 - Saturate(surface.baseColorMetallic.w)) *
        max(EyeAmbient.yzw, 0.0) / 3.14159265358979323846;
    if (validCluster && Headers[cluster].y == 0)
    {
        uint count = Headers[cluster].x;
        for (uint item = 0; item < count; ++item)
            color += EvaluateLight(surface, Lights[Indices[cluster * Grid1.w + item]]);
    }
    else
    {
        for (uint lightIndex = 0; lightIndex < Grid1.z; ++lightIndex)
            color += EvaluateLight(surface, Lights[lightIndex]);
    }

    Output[pixelIndex] = float4(color, surface.roughnessAndAlpha.y);
}
)HLSL";

struct alignas(16) GpuLight
{
    float positionRange[4]{};
    float intensity[4]{};
};

struct alignas(16) GpuSurface
{
    float positionDepth[4]{};
    float normal[4]{};
    float baseColorMetallic[4]{};
    float roughnessAndAlpha[4]{};
};

struct alignas(16) BuildParams
{
    u32 grid0[4]{};
    u32 grid1[4]{};
    float projection[4]{};
};

struct alignas(16) ShadeParams
{
    u32 grid0[4]{};
    u32 grid1[4]{};
    float depthEye[4]{};
    float eyeAmbient[4]{};
};

struct ClusterHeader
{
    u32 count = 0;
    u32 overflow = 0;
};

struct GpuResources
{
    std::array<BufferHandle, 14> buffers{};
    std::array<ShaderHandle, 2> shaders{};
    std::array<BindGroupLayoutHandle, 2> groupLayouts{};
    std::array<BindGroupHandle, 2> groups{};
    std::array<PipelineLayoutHandle, 2> pipelineLayouts{};
    std::array<PipelineHandle, 2> pipelines{};
};

[[nodiscard]] float SignedUnit(u32 value) noexcept
{
    return static_cast<float>(value & 0xFFFFU) / 32767.5f - 1.0f;
}

void PrintFailure(const char* operation, const Error& error) noexcept
{
    std::fprintf(stderr, "%s failed: %s (%s)\n", operation, cyber::core::ErrorCodeName(error.code),
                 error.message != nullptr ? error.message : "no diagnostic");
}

[[nodiscard]] Span<const Byte> AsBytes(const void* data, std::size_t size) noexcept
{
    return Span<const Byte>(static_cast<const Byte*>(data), size);
}

[[nodiscard]] bool MakeBuffer(Device& device, BufferHandle& output, cyber::rhi::BufferUsage usage,
                              u64 size, u32 stride, const char* label) noexcept
{
    cyber::rhi::BufferDesc desc{};
    desc.usage = usage;
    desc.sizeBytes = size;
    desc.elementStride = stride;
    desc.label = label;
    auto result = device.CreateBuffer(desc);
    if (!result)
    {
        PrintFailure("CreateBuffer", result.Error());
        return false;
    }
    output = result.Value();
    return true;
}

[[nodiscard]] bool MakeGroupLayout(Device& device,
                                   Span<const cyber::rhi::BindGroupLayoutEntry> entries,
                                   BindGroupLayoutHandle& output) noexcept
{
    cyber::rhi::BindGroupLayoutDesc desc{};
    desc.entries = entries;
    auto result = device.CreateBindGroupLayout(desc);
    if (!result)
    {
        PrintFailure("CreateBindGroupLayout", result.Error());
        return false;
    }
    output = result.Value();
    return true;
}

[[nodiscard]] bool MakeShader(Device& device, const char* source, cyber::rhi::ShaderStage stage,
                              ShaderHandle& output) noexcept
{
    cyber::rhi::ShaderDesc desc{};
    desc.stage = stage;
    desc.format = cyber::rhi::ShaderBytecodeFormat::HlslSource;
    desc.code = AsBytes(source, std::strlen(source));
    desc.entryPoint = "main";
    desc.label = stage == cyber::rhi::ShaderStage::Compute ? "clustered lighting compute" : nullptr;
    auto result = device.CreateShader(desc);
    if (!result)
    {
        PrintFailure("CreateShader", result.Error());
        return false;
    }
    output = result.Value();
    return true;
}

[[nodiscard]] bool MakeComputePipeline(Device& device, BindGroupLayoutHandle groupLayout,
                                       ShaderHandle shader, PipelineLayoutHandle& layoutOutput,
                                       PipelineHandle& pipelineOutput) noexcept
{
    cyber::rhi::PipelineLayoutDesc layoutDesc{};
    layoutDesc.groups[static_cast<u32>(cyber::rhi::BindGroupSlot::PerFrame)] = groupLayout;
    auto layout = device.CreatePipelineLayout(layoutDesc);
    if (!layout)
    {
        PrintFailure("CreatePipelineLayout", layout.Error());
        return false;
    }
    layoutOutput = layout.Value();

    cyber::rhi::ComputePipelineDesc pipelineDesc{};
    pipelineDesc.layout = layoutOutput;
    pipelineDesc.computeShader = shader;
    auto pipeline = device.CreateComputePipeline(pipelineDesc);
    if (!pipeline)
    {
        PrintFailure("CreateComputePipeline", pipeline.Error());
        return false;
    }
    pipelineOutput = pipeline.Value();
    return true;
}

[[nodiscard]] bool MakeBindGroup(Device& device, BindGroupLayoutHandle layout,
                                 Span<const cyber::rhi::BindingResource> resources,
                                 BindGroupHandle& output) noexcept
{
    cyber::rhi::BindGroupDesc desc{};
    desc.layout = layout;
    desc.resources = resources;
    auto result = device.CreateBindGroup(desc);
    if (!result)
    {
        PrintFailure("CreateBindGroup", result.Error());
        return false;
    }
    output = result.Value();
    return true;
}

[[nodiscard]] cyber::rhi::BindGroupLayoutEntry
LayoutEntry(u32 binding, cyber::rhi::BindingType type, u32 registerIndex) noexcept
{
    cyber::rhi::BindGroupLayoutEntry entry{};
    entry.binding = binding;
    entry.type = type;
    entry.visibility = cyber::rhi::ShaderStageBit(cyber::rhi::ShaderStage::Compute);
    entry.shaderRegister = registerIndex;
    return entry;
}

[[nodiscard]] cyber::rhi::BindingResource BufferResource(cyber::rhi::BindingType type,
                                                         BufferHandle buffer, u64 size) noexcept
{
    cyber::rhi::BindingResource resource{};
    resource.type = type;
    resource.buffer = buffer;
    resource.size = size;
    return resource;
}

[[nodiscard]] bool CreateResources(Device& device, GpuResources& resources, u64 lightBytes,
                                   u64 surfaceBytes, u64 headerBytes, u64 indexBytes,
                                   u64 outputBytes) noexcept
{
    using cyber::rhi::BindingType;
    using cyber::rhi::BufferUsage;
    constexpr u32 kUploadCount = 4;
    constexpr u32 kDestinationCount = 7;
    constexpr u32 kReadbackCount = 3;
    const u64 uploadSizes[] = {kUniformBufferBytes, kUniformBufferBytes, lightBytes, surfaceBytes};
    const u32 uploadStrides[] = {0, 0, sizeof(GpuLight), sizeof(GpuSurface)};
    const char* uploadLabels[] = {"clustered build params upload", "clustered shade params upload",
                                  "clustered lights upload", "clustered surfaces upload"};
    for (u32 index = 0; index < kUploadCount; ++index)
    {
        if (!MakeBuffer(device, resources.buffers[index],
                        BufferUsage::HostVisible | BufferUsage::CopySource, uploadSizes[index],
                        uploadStrides[index], uploadLabels[index]))
        {
            return false;
        }
    }

    const u64 destinationSizes[] = {kUniformBufferBytes, kUniformBufferBytes, lightBytes,
                                    surfaceBytes,        outputBytes,         headerBytes,
                                    indexBytes};
    const u32 destinationStrides[] = {0,
                                      0,
                                      sizeof(GpuLight),
                                      sizeof(GpuSurface),
                                      sizeof(float) * 4,
                                      sizeof(ClusterHeader),
                                      sizeof(u32)};
    const BufferUsage destinationUsages[] = {
        BufferUsage::Uniform | BufferUsage::CopyDestination,
        BufferUsage::Uniform | BufferUsage::CopyDestination,
        BufferUsage::Storage | BufferUsage::CopyDestination,
        BufferUsage::Storage | BufferUsage::CopyDestination,
        BufferUsage::Storage | BufferUsage::CopySource,
        BufferUsage::Storage | BufferUsage::CopySource,
        BufferUsage::Storage | BufferUsage::CopySource,
    };
    const char* destinationLabels[] = {"clustered build params",  "clustered shade params",
                                       "clustered lights",        "clustered surfaces",
                                       "clustered shaded output", "clustered headers",
                                       "clustered indices"};
    for (u32 index = 0; index < kDestinationCount; ++index)
    {
        if (!MakeBuffer(device, resources.buffers[kUploadCount + index], destinationUsages[index],
                        destinationSizes[index], destinationStrides[index],
                        destinationLabels[index]))
        {
            return false;
        }
    }

    const u64 readbackSizes[] = {headerBytes, indexBytes, outputBytes};
    const u32 readbackStrides[] = {sizeof(ClusterHeader), sizeof(u32), sizeof(float) * 4};
    const char* readbackLabels[] = {"clustered headers readback", "clustered indices readback",
                                    "clustered pixels readback"};
    for (u32 index = 0; index < kReadbackCount; ++index)
    {
        if (!MakeBuffer(device, resources.buffers[kUploadCount + kDestinationCount + index],
                        BufferUsage::Readback | BufferUsage::CopyDestination, readbackSizes[index],
                        readbackStrides[index], readbackLabels[index]))
        {
            return false;
        }
    }

    const std::array<cyber::rhi::BindGroupLayoutEntry, 4> buildEntries{{
        LayoutEntry(0, BindingType::UniformBuffer, 0),
        LayoutEntry(1, BindingType::ReadOnlyStorageBuffer, 0),
        LayoutEntry(2, BindingType::StorageBuffer, 0),
        LayoutEntry(3, BindingType::StorageBuffer, 1),
    }};
    if (!MakeGroupLayout(
            device,
            Span<const cyber::rhi::BindGroupLayoutEntry>(buildEntries.data(), buildEntries.size()),
            resources.groupLayouts[0]))
    {
        return false;
    }
    const std::array<cyber::rhi::BindGroupLayoutEntry, 6> shadeEntries{{
        LayoutEntry(0, BindingType::UniformBuffer, 0),
        LayoutEntry(1, BindingType::ReadOnlyStorageBuffer, 0),
        LayoutEntry(2, BindingType::ReadOnlyStorageBuffer, 1),
        LayoutEntry(3, BindingType::ReadOnlyStorageBuffer, 2),
        LayoutEntry(4, BindingType::ReadOnlyStorageBuffer, 3),
        LayoutEntry(5, BindingType::StorageBuffer, 0),
    }};
    if (!MakeGroupLayout(
            device,
            Span<const cyber::rhi::BindGroupLayoutEntry>(shadeEntries.data(), shadeEntries.size()),
            resources.groupLayouts[1]))
    {
        return false;
    }

    const std::array<cyber::rhi::BindingResource, 4> buildResources{{
        BufferResource(BindingType::UniformBuffer, resources.buffers[4], kUniformBufferBytes),
        BufferResource(BindingType::ReadOnlyStorageBuffer, resources.buffers[6], lightBytes),
        BufferResource(BindingType::StorageBuffer, resources.buffers[9], headerBytes),
        BufferResource(BindingType::StorageBuffer, resources.buffers[10], indexBytes),
    }};
    if (!MakeBindGroup(
            device, resources.groupLayouts[0],
            Span<const cyber::rhi::BindingResource>(buildResources.data(), buildResources.size()),
            resources.groups[0]))
    {
        return false;
    }
    const std::array<cyber::rhi::BindingResource, 6> shadeResources{{
        BufferResource(BindingType::UniformBuffer, resources.buffers[5], kUniformBufferBytes),
        BufferResource(BindingType::ReadOnlyStorageBuffer, resources.buffers[6], lightBytes),
        BufferResource(BindingType::ReadOnlyStorageBuffer, resources.buffers[9], headerBytes),
        BufferResource(BindingType::ReadOnlyStorageBuffer, resources.buffers[10], indexBytes),
        BufferResource(BindingType::ReadOnlyStorageBuffer, resources.buffers[7], surfaceBytes),
        BufferResource(BindingType::StorageBuffer, resources.buffers[8], outputBytes),
    }};
    if (!MakeBindGroup(
            device, resources.groupLayouts[1],
            Span<const cyber::rhi::BindingResource>(shadeResources.data(), shadeResources.size()),
            resources.groups[1]))
    {
        return false;
    }

    return MakeShader(device, kBuildShader, cyber::rhi::ShaderStage::Compute,
                      resources.shaders[0]) &&
           MakeShader(device, kShadeShader, cyber::rhi::ShaderStage::Compute,
                      resources.shaders[1]) &&
           MakeComputePipeline(device, resources.groupLayouts[0], resources.shaders[0],
                               resources.pipelineLayouts[0], resources.pipelines[0]) &&
           MakeComputePipeline(device, resources.groupLayouts[1], resources.shaders[1],
                               resources.pipelineLayouts[1], resources.pipelines[1]);
}

[[nodiscard]] bool RecordAndSubmit(Device& device, GpuResources& resources,
                                   const BuildParams& buildParams, const ShadeParams& shadeParams,
                                   Span<const GpuLight> lights, Span<const GpuSurface> surfaces,
                                   u64 headerBytes, u64 indexBytes, u64 outputBytes) noexcept
{
    using cyber::rhi::BufferBarrier;
    using cyber::rhi::BufferCopy;
    using cyber::rhi::BufferUsage;
    using cyber::rhi::QueueKind;
    using cyber::rhi::ResourceState;

    std::array<Byte, kUniformBufferBytes> buildUniform{};
    std::array<Byte, kUniformBufferBytes> shadeUniform{};
    std::memcpy(buildUniform.data(), &buildParams, sizeof(buildParams));
    std::memcpy(shadeUniform.data(), &shadeParams, sizeof(shadeParams));
    const std::array<Span<const Byte>, 4> uploads{{
        Span<const Byte>(buildUniform.data(), buildUniform.size()),
        Span<const Byte>(shadeUniform.data(), shadeUniform.size()),
        AsBytes(lights.Data(), lights.Size() * sizeof(GpuLight)),
        AsBytes(surfaces.Data(), surfaces.Size() * sizeof(GpuSurface)),
    }};
    const std::array<u64, 4> sizes{{kUniformBufferBytes, kUniformBufferBytes,
                                    lights.Size() * sizeof(GpuLight),
                                    surfaces.Size() * sizeof(GpuSurface)}};
    for (u32 index = 0; index < uploads.size(); ++index)
    {
        const auto written = device.WriteBuffer(resources.buffers[index], 0, uploads[index]);
        if (!written)
        {
            PrintFailure("WriteBuffer", written.Error());
            return false;
        }
    }

    auto created = device.CreateCommandList(
        cyber::rhi::CommandListDesc{QueueKind::Graphics, "clustered lighting sample"});
    if (!created)
    {
        PrintFailure("CreateCommandList", created.Error());
        return false;
    }
    CommandList list = std::move(created.Value());
    if (!list.Begin())
    {
        std::fprintf(stderr, "Could not begin the clustered lighting command list.\n");
        return false;
    }

    const std::array<BufferBarrier, 14> initialBarriers{{
        {resources.buffers[0], ResourceState::Undefined, ResourceState::CopySource},
        {resources.buffers[1], ResourceState::Undefined, ResourceState::CopySource},
        {resources.buffers[2], ResourceState::Undefined, ResourceState::CopySource},
        {resources.buffers[3], ResourceState::Undefined, ResourceState::CopySource},
        {resources.buffers[4], ResourceState::Undefined, ResourceState::CopyDestination},
        {resources.buffers[5], ResourceState::Undefined, ResourceState::CopyDestination},
        {resources.buffers[6], ResourceState::Undefined, ResourceState::CopyDestination},
        {resources.buffers[7], ResourceState::Undefined, ResourceState::CopyDestination},
        {resources.buffers[8], ResourceState::Undefined, ResourceState::ShaderWrite},
        {resources.buffers[9], ResourceState::Undefined, ResourceState::ShaderWrite},
        {resources.buffers[10], ResourceState::Undefined, ResourceState::ShaderWrite},
        {resources.buffers[11], ResourceState::Undefined, ResourceState::CopyDestination},
        {resources.buffers[12], ResourceState::Undefined, ResourceState::CopyDestination},
        {resources.buffers[13], ResourceState::Undefined, ResourceState::CopyDestination},
    }};
    if (auto result =
            list.Barrier(Span<const BufferBarrier>(initialBarriers.data(), initialBarriers.size()));
        !result)
    {
        PrintFailure("Initial resource barriers", result.Error());
        return false;
    }
    const std::array<BufferCopy, 4> copies{{
        {resources.buffers[0], 0, resources.buffers[4], 0, sizes[0]},
        {resources.buffers[1], 0, resources.buffers[5], 0, sizes[1]},
        {resources.buffers[2], 0, resources.buffers[6], 0, sizes[2]},
        {resources.buffers[3], 0, resources.buffers[7], 0, sizes[3]},
    }};
    for (const BufferCopy& copy : copies)
    {
        if (auto result = list.CopyBuffer(copy); !result)
        {
            PrintFailure("Upload input buffer", result.Error());
            return false;
        }
    }

    const std::array<BufferBarrier, 4> inputBarriers{{
        {resources.buffers[4], ResourceState::CopyDestination, ResourceState::ConstantBuffer},
        {resources.buffers[5], ResourceState::CopyDestination, ResourceState::ConstantBuffer},
        {resources.buffers[6], ResourceState::CopyDestination, ResourceState::ShaderRead},
        {resources.buffers[7], ResourceState::CopyDestination, ResourceState::ShaderRead},
    }};
    if (auto result =
            list.Barrier(Span<const BufferBarrier>(inputBarriers.data(), inputBarriers.size()));
        !result)
    {
        PrintFailure("Input resource barriers", result.Error());
        return false;
    }

    const cyber::rhi::DispatchArguments buildDispatch{
        (buildParams.grid1[0] * buildParams.grid1[1] * buildParams.grid0[3] + kThreadsPerGroup -
         1) /
            kThreadsPerGroup,
        1,
        1,
    };
    if (!list.SetPipeline(resources.pipelines[0]) ||
        !list.SetBindGroup(cyber::rhi::BindGroupSlot::PerFrame, resources.groups[0]) ||
        !list.Dispatch(buildDispatch))
    {
        std::fprintf(stderr, "Could not record the GPU cluster-list dispatch.\n");
        return false;
    }

    const std::array<BufferBarrier, 2> listsReady{{
        {resources.buffers[9], ResourceState::ShaderWrite, ResourceState::ShaderRead},
        {resources.buffers[10], ResourceState::ShaderWrite, ResourceState::ShaderRead},
    }};
    if (auto result = list.Barrier(Span<const BufferBarrier>(listsReady.data(), listsReady.size()));
        !result)
    {
        PrintFailure("Cluster-list barriers", result.Error());
        return false;
    }
    const cyber::rhi::DispatchArguments shadeDispatch{
        (shadeParams.grid0[0] * shadeParams.grid0[1] + kThreadsPerGroup - 1) / kThreadsPerGroup,
        1,
        1,
    };
    if (!list.SetPipeline(resources.pipelines[1]) ||
        !list.SetBindGroup(cyber::rhi::BindGroupSlot::PerFrame, resources.groups[1]) ||
        !list.Dispatch(shadeDispatch))
    {
        std::fprintf(stderr, "Could not record the GPU clustered-PBR dispatch.\n");
        return false;
    }

    const std::array<BufferBarrier, 3> outputReady{{
        {resources.buffers[9], ResourceState::ShaderRead, ResourceState::CopySource},
        {resources.buffers[10], ResourceState::ShaderRead, ResourceState::CopySource},
        {resources.buffers[8], ResourceState::ShaderWrite, ResourceState::CopySource},
    }};
    if (auto result =
            list.Barrier(Span<const BufferBarrier>(outputReady.data(), outputReady.size()));
        !result)
    {
        PrintFailure("Output resource barriers", result.Error());
        return false;
    }
    const std::array<BufferCopy, 3> readbacks{{
        {resources.buffers[9], 0, resources.buffers[11], 0, headerBytes},
        {resources.buffers[10], 0, resources.buffers[12], 0, indexBytes},
        {resources.buffers[8], 0, resources.buffers[13], 0, outputBytes},
    }};
    for (const BufferCopy& copy : readbacks)
    {
        if (auto result = list.CopyBuffer(copy); !result)
        {
            PrintFailure("Readback GPU results", result.Error());
            return false;
        }
    }
    if (auto result = list.End(); !result)
    {
        PrintFailure("End command list", result.Error());
        return false;
    }

    const auto listHandle = list.Handle();
    const cyber::rhi::SubmitDesc submit{QueueKind::Graphics,
                                        Span<const cyber::rhi::CommandListHandle>(&listHandle, 1)};
    if (auto result = device.Submit(submit); !result)
    {
        PrintFailure("Submit clustered lighting", result.Error());
        return false;
    }
    if (auto result = device.WaitIdle(); !result)
    {
        PrintFailure("WaitIdle clustered lighting", result.Error());
        return false;
    }
    return true;
}

[[nodiscard]] bool ValidateGpuResults(Device& device, const GpuResources& resources,
                                      const ClusterLightLists& cpuClusters,
                                      HeapAllocator& allocator, Span<const PbrPointLight> cpuLights,
                                      Span<const GpuSurface> surfaces,
                                      const ShadeParams& shadeParams, u64 headerBytes,
                                      u64 indexBytes, u64 outputBytes) noexcept
{
    const u32 clusterCount = cpuClusters.ClusterCount();
    cyber::memory::DynamicArray<ClusterHeader> headers(allocator);
    cyber::memory::DynamicArray<u32> indices(allocator);
    cyber::memory::DynamicArray<float> output(allocator);
    const u64 indexCapacity = static_cast<u64>(clusterCount) * kMaxLightsPerCluster;
    const u64 outputChannels = static_cast<u64>(kWidth) * kHeight * 4;
    if (headerBytes > std::numeric_limits<usize>::max() ||
        indexCapacity > std::numeric_limits<usize>::max() ||
        outputChannels > std::numeric_limits<usize>::max() || !headers.TryResize(clusterCount) ||
        !indices.TryResize(static_cast<usize>(indexCapacity)) ||
        !output.TryResize(static_cast<usize>(outputChannels)))
    {
        std::fprintf(stderr, "Could not allocate GPU result readback buffers.\n");
        return false;
    }
    const std::array<std::pair<BufferHandle, Span<Byte>>, 3> readbacks{{
        {resources.buffers[11], Span<Byte>(reinterpret_cast<Byte*>(headers.Data()),
                                           static_cast<std::size_t>(headerBytes))},
        {resources.buffers[12],
         Span<Byte>(reinterpret_cast<Byte*>(indices.Data()), static_cast<std::size_t>(indexBytes))},
        {resources.buffers[13],
         Span<Byte>(reinterpret_cast<Byte*>(output.Data()), static_cast<std::size_t>(outputBytes))},
    }};
    for (const auto& readback : readbacks)
    {
        if (auto result = device.ReadBuffer(readback.first, 0, readback.second); !result)
        {
            PrintFailure("ReadBuffer GPU result", result.Error());
            return false;
        }
    }

    u64 gpuIndexCount = 0;
    u32 maximumOccupancy = 0;
    for (u32 cluster = 0; cluster < clusterCount; ++cluster)
    {
        const ClusterHeader header = headers[cluster];
        const auto expected = cpuClusters.Lights(cluster);
        if (header.overflow != 0 || header.count > kMaxLightsPerCluster)
        {
            std::fprintf(stderr, "GPU list overflow at cluster %u (count %u, overflow %u).\n",
                         cluster, header.count, header.overflow);
            return false;
        }
        for (cyber::core::usize expectedIndex = 0; expectedIndex < expected.Size(); ++expectedIndex)
        {
            bool found = false;
            for (u32 gpuIndex = 0; gpuIndex < header.count; ++gpuIndex)
            {
                found = found ||
                        indices[static_cast<usize>(cluster) * kMaxLightsPerCluster + gpuIndex] ==
                            expected[expectedIndex];
            }
            if (!found)
            {
                std::fprintf(stderr, "GPU list omitted CPU-conservative light %u in cluster %u.\n",
                             expected[expectedIndex], cluster);
                return false;
            }
        }
        gpuIndexCount += header.count;
        maximumOccupancy = std::max(maximumOccupancy, header.count);
    }

    PbrLightingData lighting{};
    lighting.eyePosition = cyber::math::Vec3::Zero();
    lighting.ambientIrradiance = cyber::math::Vec3{0.01f, 0.012f, 0.015f};
    lighting.pointLightSpan = cpuLights;
    constexpr std::array<u32, 12> samplePixels{{
        17U,
        563U,
        1539U,
        3271U,
        6203U,
        10517U,
        18791U,
        27499U,
        39011U,
        45127U,
        51999U,
        57581U,
    }};
    float maximumError = 0.0f;
    for (u32 pixelIndex : samplePixels)
    {
        const GpuSurface& gpuSurface = surfaces[pixelIndex];
        Varyings varyings{};
        varyings.worldPosition = cyber::math::Vec3{
            gpuSurface.positionDepth[0], gpuSurface.positionDepth[1], gpuSurface.positionDepth[2]};
        varyings.normal =
            cyber::math::Vec3{gpuSurface.normal[0], gpuSurface.normal[1], gpuSurface.normal[2]};
        varyings.color = cyber::math::Color{1.0f, 1.0f, 1.0f, 1.0f};
        lighting.material.baseColor =
            cyber::math::Color{gpuSurface.baseColorMetallic[0], gpuSurface.baseColorMetallic[1],
                               gpuSurface.baseColorMetallic[2], gpuSurface.roughnessAndAlpha[1]};
        lighting.material.metallic = gpuSurface.baseColorMetallic[3];
        lighting.material.roughness = gpuSurface.roughnessAndAlpha[0];
        const cyber::math::Color expected = cyber::software::PbrPixelShader(varyings, &lighting);
        const usize channel = static_cast<usize>(pixelIndex) * 4;
        const float errors[] = {std::abs(output[channel] - expected.r),
                                std::abs(output[channel + 1] - expected.g),
                                std::abs(output[channel + 2] - expected.b)};
        for (float error : errors)
        {
            maximumError = std::max(maximumError, error);
            if (!std::isfinite(error) || error > 2.0e-3f)
            {
                std::fprintf(stderr,
                             "GPU PBR mismatch at pixel %u: CPU=(%.6f %.6f %.6f), "
                             "GPU=(%.6f %.6f %.6f).\n",
                             pixelIndex, expected.r, expected.g, expected.b, output[channel],
                             output[channel + 1], output[channel + 2]);
                return false;
            }
        }
    }

    std::printf("backend=d3d12 lights=%zu resolution=%ux%u tile=%u slices=%u clusters=%u "
                "gpu_indices=%llu max_occupancy=%u overflow=0 verified_pixels=%zu "
                "max_pbr_error=%.6f\n",
                cpuLights.Size(), shadeParams.grid0[0], shadeParams.grid0[1], kTileSize,
                kDepthSlices, clusterCount, static_cast<unsigned long long>(gpuIndexCount),
                maximumOccupancy, samplePixels.size(), maximumError);
    return true;
}

}  // namespace

int main()
{
    using cyber::rhi::DeviceDesc;
    using cyber::rhi::Feature;

    if (!cyber::rhi::IsD3D12CompiledIn())
    {
        std::fprintf(stderr, "GPU clustered lighting sample requires the D3D12 backend.\n");
        return 2;
    }

    HeapAllocator allocator;
    DeviceDesc deviceDesc{};
    deviceDesc.backend = cyber::rhi::D3D12Backend();
    deviceDesc.label = "GPU clustered lighting sample";
    auto createdDevice = Device::Create(deviceDesc, allocator);
    if (!createdDevice)
    {
        PrintFailure("Create D3D12 device", createdDevice.Error());
        return 2;
    }
    Device device = std::move(createdDevice.Value());
    if (!device.Capabilities().features.Has(Feature::Compute))
    {
        std::fprintf(stderr, "Selected D3D12 adapter does not support compute.\n");
        return 2;
    }

    ClusterGridDesc grid{};
    grid.widthPixels = kWidth;
    grid.heightPixels = kHeight;
    grid.tileSizePixels = kTileSize;
    grid.depthSlices = kDepthSlices;
    grid.verticalFovRadians = kVerticalFov;
    grid.nearPlane = kNearPlane;
    grid.farPlane = kFarPlane;
    const u32 tilesX = (kWidth + kTileSize - 1) / kTileSize;
    const u32 tilesY = (kHeight + kTileSize - 1) / kTileSize;
    const u32 clusterCount = tilesX * tilesY * kDepthSlices;
    const u64 surfaceCount = static_cast<u64>(kWidth) * kHeight;
    const u64 headerBytes = static_cast<u64>(clusterCount) * sizeof(ClusterHeader);
    const u64 indexBytes = static_cast<u64>(clusterCount) * kMaxLightsPerCluster * sizeof(u32);
    const u64 outputBytes = surfaceCount * sizeof(float) * 4;

    std::array<GpuLight, kLightCount> gpuLights{};
    std::array<PbrPointLight, kLightCount> cpuLights{};
    std::array<ClusterLightBound, kLightCount> bounds{};
    for (u32 index = 0; index < kLightCount; ++index)
    {
        const u32 hash = index * 747796405U + 2891336453U;
        PbrPointLight& light = cpuLights[index];
        light.position = cyber::math::Vec3{SignedUnit(hash) * 12.0f, SignedUnit(hash >> 7) * 8.0f,
                                           -2.0f - static_cast<float>((hash >> 16) % 46U)};
        light.intensity = cyber::math::Vec3{8.0f, 7.0f, 6.0f};
        light.range = 0.5f + static_cast<float>((hash >> 24) % 40U) / 10.0f;
        gpuLights[index] =
            GpuLight{{light.position.x, light.position.y, light.position.z, light.range},
                     {light.intensity.x, light.intensity.y, light.intensity.z, 0.0f}};
        bounds[index] =
            ClusterLightBound{light.position.x, light.position.y, -light.position.z, light.range};
    }
    ClusterLightLists cpuClusters(allocator);
    if (auto built = cyber::renderer::BuildClusterLightLists(
            Span<const ClusterLightBound>(bounds.data(), bounds.size()), grid, cpuClusters);
        !built)
    {
        PrintFailure("Build CPU cluster oracle", built.Error());
        return 1;
    }

    const float tanHalfFovY = std::tan(kVerticalFov * 0.5f);
    cyber::memory::DynamicArray<GpuSurface> surfaces(allocator);
    if (!surfaces.TryResize(static_cast<usize>(surfaceCount)))
    {
        std::fprintf(stderr, "Could not allocate synthetic clustered-light surfaces.\n");
        return 1;
    }
    for (u32 y = 0; y < kHeight; ++y)
    {
        for (u32 x = 0; x < kWidth; ++x)
        {
            const u32 index = y * kWidth + x;
            const float depth = 6.0f + static_cast<float>((x + y * 3U) % 29U) * 0.15f;
            const float ndcX = (static_cast<float>(x) + 0.5f) / kWidth * 2.0f - 1.0f;
            const float ndcY = 1.0f - (static_cast<float>(y) + 0.5f) / kHeight * 2.0f;
            GpuSurface& surface = surfaces[index];
            surface.positionDepth[0] = ndcX * depth * tanHalfFovY *
                                       (static_cast<float>(kWidth) / static_cast<float>(kHeight));
            surface.positionDepth[1] = ndcY * depth * tanHalfFovY;
            surface.positionDepth[2] = -depth;
            surface.positionDepth[3] = depth;
            surface.normal[0] = 0.1f * SignedUnit(index * 17U + 3U);
            surface.normal[1] = 0.8f;
            surface.normal[2] = 0.2f;
            surface.baseColorMetallic[0] = 0.7f;
            surface.baseColorMetallic[1] = 0.55f;
            surface.baseColorMetallic[2] = 0.35f;
            surface.baseColorMetallic[3] = 0.2f;
            surface.roughnessAndAlpha[0] = 0.65f;
            surface.roughnessAndAlpha[1] = 1.0f;
        }
    }

    BuildParams buildParams{};
    buildParams.grid0[0] = kWidth;
    buildParams.grid0[1] = kHeight;
    buildParams.grid0[2] = kTileSize;
    buildParams.grid0[3] = kDepthSlices;
    buildParams.grid1[0] = tilesX;
    buildParams.grid1[1] = tilesY;
    buildParams.grid1[2] = kLightCount;
    buildParams.grid1[3] = kMaxLightsPerCluster;
    buildParams.projection[0] = tanHalfFovY;
    buildParams.projection[1] = kNearPlane;
    buildParams.projection[2] = kFarPlane;
    buildParams.projection[3] = static_cast<float>(kWidth) / kHeight;

    ShadeParams shadeParams{};
    std::copy(std::begin(buildParams.grid0), std::end(buildParams.grid0),
              std::begin(shadeParams.grid0));
    std::copy(std::begin(buildParams.grid1), std::end(buildParams.grid1),
              std::begin(shadeParams.grid1));
    shadeParams.depthEye[0] = kNearPlane;
    shadeParams.depthEye[1] = kFarPlane;
    shadeParams.depthEye[2] = 0.0f;
    shadeParams.depthEye[3] = 0.0f;
    shadeParams.eyeAmbient[0] = 0.0f;
    shadeParams.eyeAmbient[1] = 0.01f;
    shadeParams.eyeAmbient[2] = 0.012f;
    shadeParams.eyeAmbient[3] = 0.015f;

    GpuResources resources{};
    const u64 lightBytes = sizeof(gpuLights);
    const u64 surfaceBytes = surfaces.Size() * sizeof(GpuSurface);
    if (!CreateResources(device, resources, lightBytes, surfaceBytes, headerBytes, indexBytes,
                         outputBytes))
    {
        return 1;
    }
    if (!RecordAndSubmit(device, resources, buildParams, shadeParams,
                         Span<const GpuLight>(gpuLights.data(), gpuLights.size()),
                         Span<const GpuSurface>(surfaces.Data(), surfaces.Size()), headerBytes,
                         indexBytes, outputBytes))
    {
        return 1;
    }
    return ValidateGpuResults(device, resources, cpuClusters, allocator,
                              Span<const PbrPointLight>(cpuLights.data(), cpuLights.size()),
                              Span<const GpuSurface>(surfaces.Data(), surfaces.Size()), shadeParams,
                              headerBytes, indexBytes, outputBytes)
               ? 0
               : 1;
}
