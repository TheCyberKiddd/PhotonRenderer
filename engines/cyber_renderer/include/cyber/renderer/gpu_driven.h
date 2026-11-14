#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/jobs/job_system.h>
#include <cyber/math/aabb.h>
#include <cyber/math/frustum.h>
#include <cyber/math/mat4.h>
#include <cyber/math/vec3.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/span.h>
#include <cyber/rhi/capabilities.h>
#include <cyber/rhi/command_list.h>
#include <cyber/rhi/descriptors.h>

namespace cyber::renderer
{

using cyber::core::Error;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::Result;
using cyber::core::Status;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;

// ---------------------------------------------------------------------------
// GPU-driven rendering (Phase 32)
//
// The CPU describes the scene once — instances with world bounds, draw groups
// with up to four LODs, each LOD one indexed draw — and from then on a frame
// costs the CPU a constant handful of commands whatever the instance count:
//
//   1. a compute dispatch that frustum-culls, selects a LOD, and Hi-Z
//      occlusion-culls every instance, appending survivors to per-batch
//      visible lists with atomics;
//   2. a single-group compute dispatch that compacts non-empty batches into
//      `rhi::DrawIndexedArguments` in batch order and writes the draw count;
//   3. one `DrawIndexedIndirectCount` that draws every surviving batch.
//
// `CullInstancesReference` runs the same decisions on the CPU with the same
// arithmetic, and is the oracle the GPU path is tested against. The data
// layouts below are shared with the HLSL in `gpu_driven_shaders.cpp`; their
// sizes are asserted on both sides.
// ---------------------------------------------------------------------------

/// Threads per group of the instance-culling dispatch.
inline constexpr u32 kGpuCullThreadsPerGroup = 64;
/// Threads in the single group of the compaction dispatch.
inline constexpr u32 kGpuCompactThreads = 256;
/// Threads per axis of a Hi-Z reduction group.
inline constexpr u32 kHiZThreadsPerAxis = 8;
/// Most LODs one draw group may have; matches Phase 22's scene proxies.
inline constexpr u32 kGpuDrivenMaxLods = 4;
/// Most levels a Hi-Z pyramid may have: enough for a 32768-texel edge.
inline constexpr u32 kHiZMaxLevels = 16;
/// Vertex-buffer binding the per-instance visible-ID stream is bound to by
/// `GpuCullingPipeline::RecordDraw`. The stream is one `R32Uint` per instance
/// with `VertexInputRate::PerInstance`; because instance-rate attributes are
/// offset by `firstInstance`, each indirect draw reads exactly its batch's
/// slice of the visible list.
inline constexpr u32 kGpuDrivenInstanceStreamBinding = 1;
/// Frames whose cull constants may be in flight at once. A pipeline cycles
/// through this many constant slots, so a renderer that keeps more frames in
/// flight than this needs one pipeline per extra frame.
inline constexpr u32 kGpuCullConstantSlots = 3;

/// One instance as the culling shader reads it: its world-space bounds, the
/// draw group it belongs to, and the identifier written to the visible list.
struct GpuCullInstance
{
    f32 boundsMin[3]{};
    u32 drawGroup = 0;
    f32 boundsMax[3]{};
    u32 instanceId = 0;
};
static_assert(sizeof(GpuCullInstance) == 32, "GpuCullInstance mirrors the HLSL layout");

/// A mesh with its LOD chain. LOD `l` draws batch `firstBatch + l`.
struct GpuDrawGroup
{
    u32 lodCount = 1;
    u32 firstBatch = 0;
    /// Instances of this group; the capacity of each of its batches.
    u32 instanceCount = 0;
    u32 reserved = 0;
    /// Strictly decreasing projected-radius thresholds in pixels, as Phase 22
    /// defines them. Entry `l` switches to LOD `l + 1` below it.
    f32 lodPixelThresholds[kGpuDrivenMaxLods]{};
};
static_assert(sizeof(GpuDrawGroup) == 32, "GpuDrawGroup mirrors the HLSL layout");

/// One indexed draw: a LOD of a draw group. `visibleBase` is where the
/// batch's slice of the visible list starts, and becomes the draw's
/// `firstInstance`.
struct GpuDrawBatch
{
    u32 indexCount = 0;
    u32 firstIndex = 0;
    i32 vertexOffset = 0;
    u32 visibleBase = 0;
    u32 capacity = 0;
    u32 reserved[3]{};
};
static_assert(sizeof(GpuDrawBatch) == 32, "GpuDrawBatch mirrors the HLSL layout");

/// Counters the GPU writes beside the draw count. The draw count is the first
/// 32-bit word of the same buffer, at `kGpuCullDrawCountOffset`.
struct GpuCullCounters
{
    u32 drawCount = 0;
    u32 frustumCulled = 0;
    u32 occlusionCulled = 0;
    u32 visible = 0;
    /// Survivors that found their batch full. Always zero for a finalized
    /// scene, whose batch capacities cover every instance; non-zero means the
    /// GPU and the description disagree.
    u32 overflow = 0;
    u32 reserved[3]{};
};
static_assert(sizeof(GpuCullCounters) == 32, "GpuCullCounters mirrors the HLSL layout");

/// Byte offset of the draw count inside the counter buffer.
inline constexpr u64 kGpuCullDrawCountOffset = 0;

/// What the culling pass tests. Bits combine.
inline constexpr u32 kGpuCullFrustum = 1U << 0U;
inline constexpr u32 kGpuCullOcclusion = 1U << 1U;
inline constexpr u32 kGpuCullLodSelection = 1U << 2U;
inline constexpr u32 kGpuCullAll = kGpuCullFrustum | kGpuCullOcclusion | kGpuCullLodSelection;

/// One LOD of a draw group as the caller describes it.
struct GpuDrawLod
{
    u32 indexCount = 0;
    u32 firstIndex = 0;
    i32 vertexOffset = 0;
};

/// A draw group as the caller describes it.
struct GpuDrawGroupDesc
{
    /// One to `kGpuDrivenMaxLods` LODs, most detailed first.
    cyber::memory::Span<const GpuDrawLod> lods{};
    /// `lods.Size() - 1` strictly decreasing, positive thresholds in pixels.
    f32 lodPixelThresholds[kGpuDrivenMaxLods - 1]{};
};

/// The CPU-side description of a GPU-driven scene: instances, draw groups,
/// and the batches they expand to. Built once, uploaded once; the per-frame
/// cost of culling it on the GPU does not depend on its size.
///
/// Not thread-safe. `Finalize` must be called after the last addition and
/// before the scene is culled or uploaded; adding after it un-finalizes.
class GpuDrivenScene
{
public:
    /// Constructs an empty scene whose arrays use `allocator`.
    explicit GpuDrivenScene(cyber::memory::Allocator& allocator) noexcept;
    GpuDrivenScene(const GpuDrivenScene&) = delete;
    GpuDrivenScene& operator=(const GpuDrivenScene&) = delete;
    GpuDrivenScene(GpuDrivenScene&&) noexcept = default;
    GpuDrivenScene& operator=(GpuDrivenScene&&) noexcept = default;

    /// Reserves storage for `instances` instances so that adding them does
    /// not reallocate.
    /// @return Success, or Internal when the allocation fails.
    [[nodiscard]] Status<Error> Reserve(u32 instances) noexcept;

    /// Adds a draw group and its LOD batches.
    /// @param desc One to four LODs with non-zero index counts and strictly
    /// decreasing positive thresholds.
    /// @return The group index, or InvalidArgument/Internal.
    [[nodiscard]] Result<u32, Error> AddDrawGroup(const GpuDrawGroupDesc& desc) noexcept;

    /// Adds an instance of a draw group.
    /// @param group Index returned by `AddDrawGroup`.
    /// @param worldBounds Finite, non-empty world-space bounds.
    /// @param instanceId Value the culling pass writes to the visible list.
    /// @return Success, or InvalidArgument/OutOfRange/Internal.
    [[nodiscard]] Status<Error> AddInstance(u32 group, const cyber::math::Aabb& worldBounds,
                                            u32 instanceId) noexcept;

    /// Assigns each batch its slice of the visible list. Every batch of a
    /// group gets a capacity equal to the group's instance count, because any
    /// instance may select any LOD.
    /// @return Success, or OutOfRange when the visible list would exceed
    /// 2^32 entries.
    [[nodiscard]] Status<Error> Finalize() noexcept;

    /// True after a successful `Finalize` with no addition since.
    [[nodiscard]] bool IsFinalized() const noexcept;

    [[nodiscard]] cyber::memory::Span<const GpuCullInstance> Instances() const noexcept;
    [[nodiscard]] cyber::memory::Span<const GpuDrawGroup> Groups() const noexcept;
    [[nodiscard]] cyber::memory::Span<const GpuDrawBatch> Batches() const noexcept;
    /// Entries the visible list needs: the sum of every batch's capacity.
    [[nodiscard]] u32 VisibleCapacity() const noexcept;

private:
    cyber::memory::DynamicArray<GpuCullInstance> m_instances;
    cyber::memory::DynamicArray<GpuDrawGroup> m_groups;
    cyber::memory::DynamicArray<GpuDrawBatch> m_batches;
    u32 m_visibleCapacity = 0;
    bool m_finalized = false;
};

// ---------------------------------------------------------------------------
// Hi-Z occlusion
// ---------------------------------------------------------------------------

/// One level of a Hi-Z pyramid inside its single texel array.
struct HiZLevel
{
    u32 width = 0;
    u32 height = 0;
    /// Index of the level's first texel in the pyramid array.
    u32 offset = 0;
};

/// Shape of a pyramid over a `width` x `height` depth buffer: level 0 is the
/// depth itself and level `k` is `ceil(level(k-1) / 2)` in each axis, down to
/// 1 x 1. With ceiling sizes the parent of texel `i` is always `i >> 1`, so a
/// level covers every texel of the level below without a remainder rule.
/// @param levels Receives up to `kHiZMaxLevels` levels.
/// @param texelCount Receives the total texel count of every level.
/// @return The number of levels, or 0 for an empty or oversized buffer.
[[nodiscard]] u32 ComputeHiZLayout(u32 width, u32 height, HiZLevel (&levels)[kHiZMaxLevels],
                                   u32* texelCount) noexcept;

/// A conservative depth pyramid over a reverse-Z depth buffer (near 1, far 0).
/// Each texel holds the farthest depth — the minimum — of the texels it
/// covers, so "the box is nearer than nothing here" can be answered from one
/// level with at most four reads.
class HiZPyramid
{
public:
    explicit HiZPyramid(cyber::memory::Allocator& allocator) noexcept;
    HiZPyramid(const HiZPyramid&) = delete;
    HiZPyramid& operator=(const HiZPyramid&) = delete;
    HiZPyramid(HiZPyramid&&) noexcept = default;
    HiZPyramid& operator=(HiZPyramid&&) noexcept = default;

    /// Builds every level from a depth buffer.
    /// @param depth Reverse-Z depth, row-major, top row first.
    /// @param rowPitch Floats between the starts of two rows; zero means
    /// `width`.
    /// @return Success, or InvalidArgument/OutOfRange/Internal. A failure
    /// leaves the previous pyramid unchanged.
    [[nodiscard]] Status<Error> Build(cyber::memory::Span<const f32> depth, u32 width, u32 height,
                                      u32 rowPitch = 0) noexcept;

    [[nodiscard]] bool Empty() const noexcept;
    [[nodiscard]] u32 Width() const noexcept;
    [[nodiscard]] u32 Height() const noexcept;
    [[nodiscard]] u32 LevelCount() const noexcept;
    /// The level's shape, or a zeroed value for an index out of range.
    [[nodiscard]] HiZLevel Level(u32 index) const noexcept;
    /// Every level's texels, level 0 first, in the layout the GPU uses.
    [[nodiscard]] cyber::memory::Span<const f32> Texels() const noexcept;
    /// One texel, clamped to the level's edge.
    [[nodiscard]] f32 Load(u32 level, u32 x, u32 y) const noexcept;

private:
    cyber::memory::Allocator* m_allocator = nullptr;
    cyber::memory::DynamicArray<f32> m_texels;
    HiZLevel m_levels[kHiZMaxLevels]{};
    u32 m_levelCount = 0;
};

/// Result of testing one box against a pyramid.
enum class HiZVisibility : u8
{
    /// Some part of the box may be nearer than the occluders.
    Visible = 0,
    /// The whole box is behind the farthest occluder depth it covers.
    Occluded,
    /// The box crosses the near plane or leaves the pyramid's view, so the
    /// pyramid cannot say anything about it. Treated as visible.
    Untestable,
};

/// What `TestHiZOcclusion` decided and how close it came to deciding the
/// other way. The margin lets a test of the GPU path separate a genuine
/// disagreement from one rounding step across a decision boundary.
struct HiZTestResult
{
    HiZVisibility visibility = HiZVisibility::Visible;
    /// Level the occluder depth was read from.
    u32 level = 0;
    /// Nearest depth of the box (the maximum, in reverse-Z).
    f32 boxDepth = 0.0f;
    /// Farthest occluder depth over the texels read.
    f32 occluderDepth = 0.0f;
    /// How far the depth comparison was from going the other way, relative
    /// to the occluder depth: `|boxDepth - occluderDepth| / occluderDepth`.
    f32 depthMargin = 0.0f;
    /// How far, in pixels, the box's projected edges were from a pixel
    /// boundary that `floor` would round across, or from the view edge for a
    /// box outside the view.
    f32 pixelMargin = 0.0f;
    /// Clip-space `w` of the nearest corner minus `kHiZNearGuard`; negative
    /// for a box that crosses the near plane.
    f32 nearMargin = 0.0f;
};

/// Clip-space `w` below which a corner counts as at or behind the camera.
inline constexpr f32 kHiZNearGuard = 1.0e-5f;

/// Tests a world-space box against a pyramid built from the depth that
/// `viewProjection` produced. The arithmetic is the culling shader's,
/// operation for operation.
[[nodiscard]] HiZTestResult TestHiZOcclusion(const HiZPyramid& pyramid,
                                             const cyber::math::Mat4& viewProjection,
                                             const cyber::math::Aabb& bounds) noexcept;

// ---------------------------------------------------------------------------
// The view and the reference culler
// ---------------------------------------------------------------------------

/// Everything one culling pass reads besides the scene.
struct GpuCullView
{
    /// Planes from `Frustum::FromViewProjection`, which Phase 22's
    /// `Scene::Cull` uses too.
    cyber::math::Frustum frustum{};
    cyber::math::Vec3 cameraPosition{};
    /// Viewport height divided by twice the tangent of half the vertical FOV.
    f32 projectionScalePixels = 1.0f;
    f32 nearDistance = 0.01f;
    /// The view-projection the occlusion pyramid was rendered with: the
    /// previous frame's for the usual one-frame-latency scheme.
    cyber::math::Mat4 occlusionViewProjection{};
    /// `kGpuCull*` bits.
    u32 flags = kGpuCullFrustum | kGpuCullLodSelection;
};

/// Builds a view the way Phase 22's `SceneView` is built, so the GPU culler
/// and `Scene::Cull` can be fed identical inputs.
/// @param viewProjection Reverse-Z view-projection of this frame.
/// @param verticalFovRadians Vertical field of view of the projection.
/// @param viewportHeight Height of the render target in pixels.
[[nodiscard]] GpuCullView MakeGpuCullView(const cyber::math::Mat4& viewProjection,
                                          cyber::math::Vec3 cameraPosition, f32 verticalFovRadians,
                                          u32 viewportHeight, f32 nearDistance) noexcept;

/// The decision for one instance, with the closest it came to another.
struct GpuInstanceDecision
{
    bool visible = false;
    bool frustumCulled = false;
    bool occlusionCulled = false;
    u32 lod = 0;
    /// Batch the instance is drawn by when visible.
    u32 batch = 0;
    /// Smallest |plane distance| among the planes that decided the frustum
    /// test, in world units.
    f32 frustumMargin = 0.0f;
    /// Smallest relative distance between the projected radius and a LOD
    /// threshold, or a large value when the group has one LOD.
    f32 lodMargin = 0.0f;
    HiZTestResult occlusion{};
};

/// Decides one instance exactly as the culling shader does.
/// @param occluders Pyramid for occlusion culling, or null to skip it.
[[nodiscard]] GpuInstanceDecision DecideGpuInstance(const GpuCullInstance& instance,
                                                    const GpuDrawGroup& group,
                                                    const GpuCullView& view,
                                                    const HiZPyramid* occluders) noexcept;

/// What a culling pass produces, read back or computed on the CPU.
struct GpuCullOutput
{
    explicit GpuCullOutput(cyber::memory::Allocator& allocator) noexcept;

    /// Allocator of every array below and of scratch storage.
    cyber::memory::Allocator* allocator = nullptr;
    /// Survivors per batch.
    cyber::memory::DynamicArray<u32> batchCounts;
    /// `VisibleCapacity()` entries; batch `b` owns
    /// `[visibleBase, visibleBase + batchCounts[b])`. The GPU appends with
    /// atomics, so the order inside one batch is unspecified there and
    /// ascending instance order here.
    cyber::memory::DynamicArray<u32> visibleInstances;
    /// The compacted draws, in batch order on both paths.
    cyber::memory::DynamicArray<cyber::rhi::DrawIndexedArguments> draws;
    GpuCullCounters counters{};
};

/// Culls a finalized scene on the CPU with the culling shader's arithmetic.
/// @param occluders Pyramid for `kGpuCullOcclusion`, or null.
/// @param jobs Optional running job system; results are identical with and
/// without it.
/// @return Success, or InvalidArgument for an unfinalized scene or a view that
/// asks for occlusion without a pyramid, or Internal on allocation failure.
[[nodiscard]] Status<Error> CullInstancesReference(const GpuDrivenScene& scene,
                                                   const GpuCullView& view,
                                                   const HiZPyramid* occluders,
                                                   GpuCullOutput& output,
                                                   cyber::jobs::JobSystem* jobs = nullptr) noexcept;

// ---------------------------------------------------------------------------
// Geometry path: mesh shaders where supported
// ---------------------------------------------------------------------------

/// Whether this build contains a mesh-shader pipeline.
///
/// Mesh and amplification shaders need shader model 6.5, which only DXC
/// compiles; the RHI compiles HLSL with `D3DCompile`, whose highest model is
/// 5.1 (ADR 0019). The meshlet data and the amplification stage's culling are
/// built and tested on the CPU (`meshlet.h`), and a capable adapter resolves
/// to the vertex pipeline with `GeometryFallbackReason::NotImplemented`, so a
/// log line separates "this GPU cannot" from "this build does not", as the
/// ray-tracing gate does.
inline constexpr bool kMeshShaderPathCompiledIn = false;

/// The pipeline that turns meshes into triangles.
enum class GeometryPath : u8
{
    /// Input assembler and vertex shader, fed by indirect draws.
    VertexPipeline = 0,
    /// Amplification and mesh shaders over meshlets.
    MeshShader,
};

/// What the caller asks for.
enum class GeometryPathPreference : u8
{
    /// Always the vertex pipeline.
    VertexOnly = 0,
    /// Mesh shaders wherever the adapter and the build support them.
    PreferMeshShaders,
};

/// Why the vertex pipeline was chosen. `None` means the preference was met.
enum class GeometryFallbackReason : u8
{
    None = 0,
    /// The preference is `VertexOnly`.
    PolicyDisabled,
    /// The adapter does not report `rhi::Feature::MeshShaders`.
    Unsupported,
    /// The adapter is capable but this build has no mesh-shader pipeline; see
    /// `kMeshShaderPathCompiledIn`.
    NotImplemented,
};

/// The resolved geometry path.
struct GeometryPathPlan
{
    GeometryPath path = GeometryPath::VertexPipeline;
    GeometryFallbackReason reason = GeometryFallbackReason::None;
    /// What the adapter reported, kept for logs.
    bool meshShadersReported = false;
};

/// Resolves the geometry path. Pure: depends only on its arguments.
[[nodiscard]] GeometryPathPlan SelectGeometryPath(const cyber::rhi::Capabilities& capabilities,
                                                  GeometryPathPreference preference) noexcept;

[[nodiscard]] const char* GeometryPathName(GeometryPath path) noexcept;
[[nodiscard]] const char* GeometryFallbackReasonName(GeometryFallbackReason reason) noexcept;

// ---------------------------------------------------------------------------
// Async compute
// ---------------------------------------------------------------------------

/// What the caller asks for.
enum class AsyncComputePreference : u8
{
    /// Culling and light culling run on the graphics queue.
    Off = 0,
    /// Run them on the compute queue only when a measurement shows a gain.
    /// Async compute helps on some GPUs and costs on others, so `Auto` never
    /// keeps it on faith.
    Auto,
    /// Run them on the compute queue whenever one exists, for measuring.
    Force,
};

/// Why the plan is what it is. `None` means async compute was selected.
enum class AsyncComputeReason : u8
{
    None = 0,
    /// The preference is `Off`.
    PolicyDisabled,
    /// The device has no compute queue.
    NoComputeQueue,
    /// `Auto` without enough samples to decide.
    NotMeasured,
    /// `Auto`, and the measured gain is below the policy's threshold.
    NoMeasuredBenefit,
};

/// Timings of the same frame run both ways, from GPU timestamps or wall-clock
/// time around a fence wait.
struct AsyncComputeMeasurement
{
    /// Median frame time with every pass on the graphics queue.
    f32 serialMilliseconds = 0.0f;
    /// Median frame time with culling on the compute queue.
    f32 asyncMilliseconds = 0.0f;
    /// Frames behind each median.
    u32 samples = 0;
};

/// Policy for `SelectAsyncComputePlan`.
struct AsyncComputePolicyDesc
{
    AsyncComputePreference preference = AsyncComputePreference::Auto;
    /// Smallest fraction of the serial frame time the async path must save to
    /// be kept under `Auto`. Below it the extra fence is not worth having.
    f32 minimumGainFraction = 0.05f;
    /// Samples each median needs before `Auto` trusts it.
    u32 minimumSamples = 8;
};

/// The resolved queue assignment.
struct AsyncComputePlan
{
    bool useAsyncCompute = false;
    /// Queue the instance-culling and compaction passes are recorded for.
    cyber::rhi::QueueKind cullingQueue = cyber::rhi::QueueKind::Graphics;
    /// Queue the light-culling pass is recorded for.
    cyber::rhi::QueueKind lightCullingQueue = cyber::rhi::QueueKind::Graphics;
    AsyncComputeReason reason = AsyncComputeReason::None;
    /// `(serial - async) / serial` from the measurement, or 0 without one.
    f32 measuredGainFraction = 0.0f;
};

/// Resolves the queue assignment. Pure: depends only on its arguments.
/// @param computeQueueAvailable Whether `Device::Queue(QueueKind::Compute)` is
/// valid.
/// @param measurement Timings for `Auto`, or null.
[[nodiscard]] AsyncComputePlan
SelectAsyncComputePlan(const AsyncComputePolicyDesc& policy, bool computeQueueAvailable,
                       const AsyncComputeMeasurement* measurement) noexcept;

[[nodiscard]] const char* AsyncComputeReasonName(AsyncComputeReason reason) noexcept;

}  // namespace cyber::renderer
