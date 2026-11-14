#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/span.h>
#include <cyber/renderer/gpu_driven.h>
#include <cyber/rhi/command_list.h>
#include <cyber/rhi/device.h>
#include <cyber/rhi/handle.h>
#include <cyber/rhi/resource_state.h>

namespace cyber::renderer
{

/// Capacities a culling pipeline is created for. Buffers are sized once; a
/// scene larger than any capacity is refused at upload rather than truncated.
struct GpuCullingPipelineDesc
{
    u32 maxInstances = 0;
    u32 maxGroups = 0;
    u32 maxBatches = 0;
    /// Entries of the visible list: at least the scene's `VisibleCapacity()`.
    u32 maxVisible = 0;
    /// Largest depth buffer a Hi-Z pyramid is built over; zero disables
    /// occlusion culling for this pipeline.
    u32 hiZMaxWidth = 0;
    u32 hiZMaxHeight = 0;
    /// Optional prefix for buffer names in debuggers. Borrowed.
    const char* label = nullptr;
};

/// Recording-time counters for one pipeline, for tests and profiling.
struct GpuCullingPipelineStats
{
    /// Dispatches the last `RecordCull` recorded.
    u32 cullDispatches = 0;
    /// Groups of the instance-culling dispatch.
    u32 cullGroupsX = 0;
    u32 cullGroupsY = 0;
    /// Commands the last `RecordDraw` recorded: always one indirect draw.
    u32 drawCalls = 0;
};

/// The GPU side of GPU-driven rendering: buffers, the three compute pipelines
/// (Hi-Z reduction, instance culling, compaction), and the commands that run
/// them. `GpuDrivenScene` and `GpuCullView` are its inputs, and
/// `CullInstancesReference` the oracle its output is tested against.
///
/// Queues: `RecordSceneUpload`, `RecordHiZUpload`, `RecordHiZBuild`, and
/// `RecordCull` record on a graphics or compute list; `RecordAcquireForDraw`
/// and `RecordDraw` on a graphics list. Running the first group on the
/// compute queue is async compute: the caller signals a fence after the
/// compute submission and makes the graphics submission wait for it with an
/// `rhi::GpuWait`, which is the only ordering between queues the RHI has.
///
/// Resource states are tracked here in recording order, so lists must be
/// submitted in the order they were recorded. Not thread-safe.
class GpuCullingPipeline
{
public:
    /// Constructs an empty pipeline that records nothing.
    GpuCullingPipeline() noexcept = default;
    /// Destroys every RHI object the pipeline created. The caller must have
    /// waited for the GPU to finish with them.
    ~GpuCullingPipeline();
    GpuCullingPipeline(const GpuCullingPipeline&) = delete;
    GpuCullingPipeline& operator=(const GpuCullingPipeline&) = delete;
    GpuCullingPipeline(GpuCullingPipeline&& other) noexcept;
    GpuCullingPipeline& operator=(GpuCullingPipeline&& other) noexcept;

    /// Creates the buffers, shaders, layouts, and pipelines.
    /// @param device Device that must outlive the pipeline and report
    /// `Feature::Compute` and `Feature::IndirectDrawCount`.
    /// @return A pipeline, or Unsupported/InvalidArgument/Internal.
    [[nodiscard]] static Result<GpuCullingPipeline, Error>
    Create(cyber::rhi::Device& device, const GpuCullingPipelineDesc& desc,
           cyber::memory::Allocator& allocator) noexcept;

    /// Writes a finalized scene to staging memory and records the copies into
    /// the device buffers. The staging memory is reused, so the previous
    /// upload's list must have finished on the GPU.
    [[nodiscard]] Status<Error> RecordSceneUpload(cyber::rhi::CommandList& list,
                                                  const GpuDrivenScene& scene) noexcept;

    /// Uploads a pyramid built on the CPU, for occlusion culling against a
    /// depth buffer the CPU owns (the software renderer's, or a test's).
    [[nodiscard]] Status<Error> RecordHiZUpload(cyber::rhi::CommandList& list,
                                                const HiZPyramid& pyramid) noexcept;

    /// Builds the pyramid on the GPU from a depth buffer already in GPU
    /// memory: level 0 is copied, then each level is reduced from the one
    /// below with a UAV barrier between dispatches.
    /// @param depth Structured buffer of 32-bit floats with storage usage, in
    /// `ShaderRead` state.
    /// @param rowPitch Floats between rows of `depth`; zero means `width`.
    [[nodiscard]] Status<Error> RecordHiZBuild(cyber::rhi::CommandList& list,
                                               cyber::rhi::BufferHandle depth, u32 width,
                                               u32 height, u32 rowPitch = 0) noexcept;

    /// Records one frame's culling: counter clears, the instance-culling
    /// dispatch, and the compaction dispatch. Requires a scene upload, and a
    /// Hi-Z upload or build when the view asks for occlusion.
    [[nodiscard]] Status<Error> RecordCull(cyber::rhi::CommandList& list,
                                           const GpuCullView& view) noexcept;

    /// Transitions the draw arguments and draw count to `IndirectArgument`
    /// and the visible list to `VertexBuffer`. Record it on the graphics list
    /// that draws, after any cross-queue wait.
    [[nodiscard]] Status<Error> RecordAcquireForDraw(cyber::rhi::CommandList& list) noexcept;

    /// Binds the visible list at `kGpuDrivenInstanceStreamBinding` and records
    /// one `DrawIndexedIndirectCount` for every batch. The caller has begun a
    /// pass and bound a pipeline whose vertex layout declares that binding as
    /// a per-instance `R32Uint` attribute, plus the shared index buffer and
    /// any per-vertex streams.
    [[nodiscard]] Status<Error> RecordDraw(cyber::rhi::CommandList& list) noexcept;

    /// Copies the counters, per-batch counts, visible list, and draws into
    /// readback memory. Read them with `ReadBack` once the list has finished.
    [[nodiscard]] Status<Error> RecordReadback(cyber::rhi::CommandList& list) noexcept;

    /// Reads what the last `RecordReadback` copied, sized to the uploaded
    /// scene, into `output`.
    [[nodiscard]] Status<Error> ReadBack(GpuCullOutput& output) noexcept;

    /// Reads the last Hi-Z pyramid copied by `RecordReadback`, as texels in
    /// `HiZPyramid::Texels` order, for comparing a GPU build with the CPU's.
    [[nodiscard]] Status<Error> ReadBackHiZ(cyber::memory::Span<f32> texels) noexcept;

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] const GpuCullingPipelineStats& Stats() const noexcept;
    [[nodiscard]] cyber::rhi::BufferHandle ArgumentBuffer() const noexcept;
    [[nodiscard]] cyber::rhi::BufferHandle CounterBuffer() const noexcept;
    [[nodiscard]] cyber::rhi::BufferHandle VisibleBuffer() const noexcept;
    /// Batches of the uploaded scene: the `maxDrawCount` of the indirect draw.
    [[nodiscard]] u32 BatchCount() const noexcept;

private:
    struct Impl;
    explicit GpuCullingPipeline(Impl* impl) noexcept;
    void Destroy() noexcept;
    Impl* m_impl = nullptr;
};

}  // namespace cyber::renderer
