#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/span.h>
#include <cyber/renderer/clustered_lighting.h>
#include <cyber/rhi/command_list.h>
#include <cyber/rhi/device.h>
#include <cyber/rhi/handle.h>

namespace cyber::renderer
{

using cyber::core::Error;
using cyber::core::Result;
using cyber::core::Status;
using cyber::core::u32;

/// Capacities a light-culling pass is created for (Phase 32).
struct GpuLightCullingDesc
{
    /// The cluster grid, as `BuildClusterLightLists` takes it.
    ClusterGridDesc grid{};
    u32 maxLights = 0;
    /// Light slots per cluster. A cluster with more lights keeps the first
    /// `maxLightsPerCluster` in input order and reports the overflow.
    u32 maxLightsPerCluster = 128;
};

/// One cluster's list as the GPU wrote it.
struct GpuClusterHeader
{
    u32 count = 0;
    /// Non-zero when the cluster had more lights than slots.
    u32 overflow = 0;
};

/// Cluster light lists built by a compute shader, so light culling can run on
/// the async compute queue beside instance culling. The lists are
/// conservative against `BuildClusterLightLists`: every light the CPU lists
/// for a cluster is in the GPU's list too. The GPU's slice range is widened
/// by one slice each way so that its single-precision logarithm can never
/// drop a light the CPU's double-precision one keeps; the CPU stays the
/// tight reference.
///
/// Records on a graphics or compute list. Resource states are tracked in
/// recording order, as `GpuCullingPipeline` does. Not thread-safe.
class GpuLightCullingPass
{
public:
    GpuLightCullingPass() noexcept = default;
    /// Destroys every RHI object the pass created; the caller must have
    /// waited for the GPU to finish with them.
    ~GpuLightCullingPass();
    GpuLightCullingPass(const GpuLightCullingPass&) = delete;
    GpuLightCullingPass& operator=(const GpuLightCullingPass&) = delete;
    GpuLightCullingPass(GpuLightCullingPass&& other) noexcept;
    GpuLightCullingPass& operator=(GpuLightCullingPass&& other) noexcept;

    /// Creates the buffers and the compute pipeline.
    /// @return A pass, or Unsupported/InvalidArgument/Internal.
    [[nodiscard]] static Result<GpuLightCullingPass, Error>
    Create(cyber::rhi::Device& device, const GpuLightCullingDesc& desc,
           cyber::memory::Allocator& allocator) noexcept;

    /// Writes view-space light bounds to staging memory and records their
    /// copy. The previous upload's list must have finished on the GPU.
    [[nodiscard]] Status<Error>
    RecordLightUpload(cyber::rhi::CommandList& list,
                      cyber::memory::Span<const ClusterLightBound> lights) noexcept;

    /// Records the light-list dispatch, one thread per cluster.
    [[nodiscard]] Status<Error> RecordCull(cyber::rhi::CommandList& list) noexcept;

    /// Copies headers and indices into readback memory.
    [[nodiscard]] Status<Error> RecordReadback(cyber::rhi::CommandList& list) noexcept;

    /// Reads what the last `RecordReadback` copied: one header per cluster and
    /// `maxLightsPerCluster` index slots per cluster.
    [[nodiscard]] Status<Error> ReadBack(cyber::memory::DynamicArray<GpuClusterHeader>& headers,
                                         cyber::memory::DynamicArray<u32>& indices) noexcept;

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] u32 ClusterCount() const noexcept;
    [[nodiscard]] u32 MaxLightsPerCluster() const noexcept;
    /// Per-cluster `GpuClusterHeader`s, for a shading pass to bind.
    [[nodiscard]] cyber::rhi::BufferHandle HeaderBuffer() const noexcept;
    /// Per-cluster light index slots, for a shading pass to bind.
    [[nodiscard]] cyber::rhi::BufferHandle IndexBuffer() const noexcept;

private:
    struct Impl;
    explicit GpuLightCullingPass(Impl* impl) noexcept;
    void Destroy() noexcept;
    Impl* m_impl = nullptr;
};

}  // namespace cyber::renderer
