#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/renderer/render_graph.h>
#include <cyber/rhi/device.h>
#include <cyber/rhi/render_pass.h>
#include <cyber/rhi/sampler.h>

namespace cyber::renderer
{

/// Caller-owned data for one reverse-Z depth-only shadow pass. `depthTexture`
/// must be the texture imported as `depthResource`, and `depthView` must be a
/// depth-only view of that texture. The descriptor must remain alive until the
/// graph has finished executing the pass.
struct ShadowDepthPassDesc
{
    const char* name = "Shadow depth";
    GraphResourceHandle depthResource{};
    cyber::rhi::TextureHandle depthTexture{};
    cyber::rhi::TextureViewHandle depthView{};
    RenderPassExecute record = nullptr;
    void* userData = nullptr;
};

/// Caller-owned data for a pass that samples one or more imported shadow maps.
/// Each map is transitioned to ShaderRead and graph dependencies ensure its
/// depth pass completes first. The descriptor and resource span must remain
/// alive until graph execution finishes.
struct ShadowSamplingPassDesc
{
    const char* name = "Shadow sampling";
    cyber::memory::Span<const GraphResourceHandle> shadowResources{};
    RenderPassExecute record = nullptr;
    void* userData = nullptr;
};

/// Owned GPU storage for one reverse-Z depth map, with separate depth-attachment
/// and sampled views. Import it once per frame graph before scheduling passes.
struct ShadowMapResources
{
    cyber::rhi::TextureHandle depthTexture{};
    cyber::rhi::TextureViewHandle depthView{};
    cyber::rhi::TextureViewHandle sampledView{};
    cyber::core::u32 width = 0;
    cyber::core::u32 height = 0;
};

/// Creates a retained D32 depth texture with depth-write and shader-sampling
/// usage plus distinct DSV and sampled views. `output` must be empty; the caller
/// owns the returned handles.
[[nodiscard]] Status<Error> CreateShadowMapResources(cyber::rhi::Device& device,
                                                     cyber::core::u32 width,
                                                     cyber::core::u32 height,
                                                     ShadowMapResources& output) noexcept;

/// Imports a caller-owned shadow map into a frame graph. Destroy the graph
/// before destroying the map resources.
[[nodiscard]] Result<GraphResourceHandle, Error>
ImportShadowMap(RenderGraph& graph, const char* name,
                const ShadowMapResources& resources) noexcept;

/// Destroys the map view and texture, clearing their handles on success.
[[nodiscard]] Status<Error> DestroyShadowMapResources(cyber::rhi::Device& device,
                                                      ShadowMapResources& resources) noexcept;

/// Adds a graph pass that transitions its depth map to DepthWrite, clears to
/// reverse-Z far depth 0, opens a depth-only render pass, and invokes `record`
/// while that pass is active. The callback owns pipeline binding and draw
/// commands, including a GreaterEqual depth test.
[[nodiscard]] Status<Error> AddShadowDepthPass(RenderGraph& graph,
                                               ShadowDepthPassDesc& desc) noexcept;

/// Adds a graph root that declares every listed shadow map for shader sampling.
/// This makes the depth-write-to-shader-read transitions and producer
/// dependencies explicit while leaving pipeline binding to the caller.
[[nodiscard]] Status<Error> AddShadowSamplingPass(RenderGraph& graph,
                                                  ShadowSamplingPassDesc& desc) noexcept;

/// Creates the comparison sampler convention for reverse-Z shadow maps.
/// A sampled depth less than or equal to the receiver is lit. Zero-depth
/// (opaque-black) border samples keep lookups outside the map lit.
[[nodiscard]] cyber::rhi::SamplerDesc MakeReverseZShadowSamplerDesc() noexcept;

}  // namespace cyber::renderer
