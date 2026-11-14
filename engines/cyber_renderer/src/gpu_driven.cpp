#include <cyber/math/vec4.h>
#include <cyber/renderer/gpu_driven.h>

#include <cmath>
#include <limits>

namespace cyber::renderer
{

namespace
{

using cyber::core::ErrorCode;
using cyber::core::usize;
using cyber::math::Aabb;
using cyber::math::Mat4;
using cyber::math::Vec3;
using cyber::math::Vec4;
using cyber::memory::Span;

[[nodiscard]] Status<Error> Fail(ErrorCode code, const char* message) noexcept
{
    return Status<Error>::Err(Error{code, message});
}

[[nodiscard]] bool IsFinite(f32 value) noexcept
{
    return std::isfinite(value);
}

[[nodiscard]] bool IsFinite(Vec3 value) noexcept
{
    return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
}

[[nodiscard]] f32 Min(f32 lhs, f32 rhs) noexcept
{
    return rhs < lhs ? rhs : lhs;
}

[[nodiscard]] f32 Max(f32 lhs, f32 rhs) noexcept
{
    return rhs > lhs ? rhs : lhs;
}

[[nodiscard]] f32 Abs(f32 value) noexcept
{
    return value < 0.0f ? -value : value;
}

// Distance from a pixel coordinate to the nearest integer: how far `floor`
// is from returning a different texel.
[[nodiscard]] f32 FloorMargin(f32 value) noexcept
{
    const f32 fraction = value - std::floor(value);
    return Min(fraction, 1.0f - fraction);
}

inline constexpr f32 kLargeMargin = std::numeric_limits<f32>::max();

}  // namespace

// ---------------------------------------------------------------------------
// GpuDrivenScene
// ---------------------------------------------------------------------------

GpuDrivenScene::GpuDrivenScene(cyber::memory::Allocator& allocator) noexcept
    : m_instances(allocator), m_groups(allocator), m_batches(allocator)
{
}

Status<Error> GpuDrivenScene::Reserve(u32 instances) noexcept
{
    if (!m_instances.TryReserve(instances))
    {
        return Fail(ErrorCode::Internal, "gpu-driven scene: instance reservation failed");
    }
    return Status<Error>::Ok();
}

Result<u32, Error> GpuDrivenScene::AddDrawGroup(const GpuDrawGroupDesc& desc) noexcept
{
    using ResultType = Result<u32, Error>;
    const usize lodCount = desc.lods.Size();
    if (lodCount == 0 || lodCount > kGpuDrivenMaxLods)
    {
        return ResultType::Err(
            Error{ErrorCode::InvalidArgument, "gpu-driven scene: a draw group needs 1-4 LODs"});
    }
    for (usize lod = 0; lod < lodCount; ++lod)
    {
        if (desc.lods[lod].indexCount == 0)
        {
            return ResultType::Err(
                Error{ErrorCode::InvalidArgument, "gpu-driven scene: a LOD draws no indices"});
        }
    }
    for (usize index = 0; index + 1 < lodCount; ++index)
    {
        const f32 threshold = desc.lodPixelThresholds[index];
        if (!IsFinite(threshold) || threshold <= 0.0f ||
            (index > 0 && desc.lodPixelThresholds[index - 1] <= threshold))
        {
            return ResultType::Err(Error{ErrorCode::InvalidArgument,
                                         "gpu-driven scene: LOD thresholds must be positive and "
                                         "strictly decreasing"});
        }
    }
    if (m_groups.Size() >= std::numeric_limits<u32>::max() ||
        m_batches.Size() + lodCount > std::numeric_limits<u32>::max())
    {
        return ResultType::Err(Error{ErrorCode::OutOfRange, "gpu-driven scene: too many groups"});
    }

    GpuDrawGroup group{};
    group.lodCount = static_cast<u32>(lodCount);
    group.firstBatch = static_cast<u32>(m_batches.Size());
    for (usize index = 0; index + 1 < lodCount; ++index)
    {
        group.lodPixelThresholds[index] = desc.lodPixelThresholds[index];
    }
    if (!m_batches.TryReserve(m_batches.Size() + lodCount) || !m_groups.TryPushBack(group))
    {
        return ResultType::Err(Error{ErrorCode::Internal, "gpu-driven scene: allocation failed"});
    }
    for (usize lod = 0; lod < lodCount; ++lod)
    {
        GpuDrawBatch batch{};
        batch.indexCount = desc.lods[lod].indexCount;
        batch.firstIndex = desc.lods[lod].firstIndex;
        batch.vertexOffset = desc.lods[lod].vertexOffset;
        // Reserved above, so this cannot fail.
        (void)m_batches.TryPushBack(batch);
    }
    m_finalized = false;
    return ResultType::Ok(static_cast<u32>(m_groups.Size() - 1));
}

Status<Error> GpuDrivenScene::AddInstance(u32 group, const Aabb& worldBounds,
                                          u32 instanceId) noexcept
{
    if (group >= m_groups.Size())
    {
        return Fail(ErrorCode::InvalidArgument, "gpu-driven scene: unknown draw group");
    }
    if (worldBounds.IsEmpty() || !IsFinite(worldBounds.Min()) || !IsFinite(worldBounds.Max()))
    {
        return Fail(ErrorCode::InvalidArgument,
                    "gpu-driven scene: instance bounds must be finite and non-empty");
    }
    if (m_instances.Size() >= std::numeric_limits<u32>::max())
    {
        return Fail(ErrorCode::OutOfRange, "gpu-driven scene: too many instances");
    }
    GpuCullInstance instance{};
    const Vec3 minimum = worldBounds.Min();
    const Vec3 maximum = worldBounds.Max();
    instance.boundsMin[0] = minimum.x;
    instance.boundsMin[1] = minimum.y;
    instance.boundsMin[2] = minimum.z;
    instance.boundsMax[0] = maximum.x;
    instance.boundsMax[1] = maximum.y;
    instance.boundsMax[2] = maximum.z;
    instance.drawGroup = group;
    instance.instanceId = instanceId;
    if (!m_instances.TryPushBack(instance))
    {
        return Fail(ErrorCode::Internal, "gpu-driven scene: instance allocation failed");
    }
    m_finalized = false;
    return Status<Error>::Ok();
}

Status<Error> GpuDrivenScene::Finalize() noexcept
{
    for (usize index = 0; index < m_groups.Size(); ++index)
    {
        m_groups[index].instanceCount = 0;
    }
    for (usize index = 0; index < m_instances.Size(); ++index)
    {
        ++m_groups[m_instances[index].drawGroup].instanceCount;
    }
    u64 running = 0;
    for (usize groupIndex = 0; groupIndex < m_groups.Size(); ++groupIndex)
    {
        const GpuDrawGroup& group = m_groups[groupIndex];
        for (u32 lod = 0; lod < group.lodCount; ++lod)
        {
            GpuDrawBatch& batch = m_batches[group.firstBatch + lod];
            batch.visibleBase = static_cast<u32>(running);
            batch.capacity = group.instanceCount;
            running += group.instanceCount;
            if (running > std::numeric_limits<u32>::max())
            {
                m_finalized = false;
                return Fail(ErrorCode::OutOfRange,
                            "gpu-driven scene: the visible list would exceed 2^32 entries");
            }
        }
    }
    m_visibleCapacity = static_cast<u32>(running);
    m_finalized = true;
    return Status<Error>::Ok();
}

bool GpuDrivenScene::IsFinalized() const noexcept
{
    return m_finalized;
}

Span<const GpuCullInstance> GpuDrivenScene::Instances() const noexcept
{
    return Span<const GpuCullInstance>(m_instances.Data(), m_instances.Size());
}

Span<const GpuDrawGroup> GpuDrivenScene::Groups() const noexcept
{
    return Span<const GpuDrawGroup>(m_groups.Data(), m_groups.Size());
}

Span<const GpuDrawBatch> GpuDrivenScene::Batches() const noexcept
{
    return Span<const GpuDrawBatch>(m_batches.Data(), m_batches.Size());
}

u32 GpuDrivenScene::VisibleCapacity() const noexcept
{
    return m_visibleCapacity;
}

// ---------------------------------------------------------------------------
// Hi-Z pyramid
// ---------------------------------------------------------------------------

u32 ComputeHiZLayout(u32 width, u32 height, HiZLevel (&levels)[kHiZMaxLevels],
                     u32* texelCount) noexcept
{
    if (texelCount != nullptr)
    {
        *texelCount = 0;
    }
    if (width == 0 || height == 0)
    {
        return 0;
    }
    u64 offset = 0;
    u32 count = 0;
    u32 levelWidth = width;
    u32 levelHeight = height;
    while (true)
    {
        if (count >= kHiZMaxLevels)
        {
            return 0;
        }
        levels[count].width = levelWidth;
        levels[count].height = levelHeight;
        levels[count].offset = static_cast<u32>(offset);
        offset += static_cast<u64>(levelWidth) * levelHeight;
        if (offset > std::numeric_limits<u32>::max())
        {
            return 0;
        }
        ++count;
        if (levelWidth == 1 && levelHeight == 1)
        {
            break;
        }
        levelWidth = (levelWidth + 1) / 2;
        levelHeight = (levelHeight + 1) / 2;
    }
    if (texelCount != nullptr)
    {
        *texelCount = static_cast<u32>(offset);
    }
    return count;
}

HiZPyramid::HiZPyramid(cyber::memory::Allocator& allocator) noexcept
    : m_allocator(&allocator), m_texels(allocator)
{
}

Status<Error> HiZPyramid::Build(Span<const f32> depth, u32 width, u32 height, u32 rowPitch) noexcept
{
    const u32 pitch = rowPitch == 0 ? width : rowPitch;
    if (width == 0 || height == 0 || pitch < width)
    {
        return Fail(ErrorCode::InvalidArgument, "hi-z: the depth buffer has no texels");
    }
    if (static_cast<u64>(pitch) * (height - 1) + width > depth.Size())
    {
        return Fail(ErrorCode::InvalidArgument, "hi-z: the depth span is smaller than its shape");
    }
    HiZLevel levels[kHiZMaxLevels]{};
    u32 texelCount = 0;
    const u32 levelCount = ComputeHiZLayout(width, height, levels, &texelCount);
    if (levelCount == 0)
    {
        return Fail(ErrorCode::OutOfRange, "hi-z: the depth buffer is too large for a pyramid");
    }
    cyber::memory::DynamicArray<f32> texels(*m_allocator);
    if (!texels.TryResize(texelCount))
    {
        return Fail(ErrorCode::Internal, "hi-z: pyramid allocation failed");
    }
    for (u32 y = 0; y < height; ++y)
    {
        for (u32 x = 0; x < width; ++x)
        {
            texels[static_cast<usize>(y) * width + x] = depth[static_cast<usize>(y) * pitch + x];
        }
    }
    for (u32 level = 1; level < levelCount; ++level)
    {
        const HiZLevel& source = levels[level - 1];
        const HiZLevel& destination = levels[level];
        for (u32 y = 0; y < destination.height; ++y)
        {
            for (u32 x = 0; x < destination.width; ++x)
            {
                // Ceiling sizes put 2x inside the source; 2x + 1 may not be.
                const u32 x0 = 2 * x;
                const u32 y0 = 2 * y;
                const u32 x1 = x0 + 1 < source.width ? x0 + 1 : x0;
                const u32 y1 = y0 + 1 < source.height ? y0 + 1 : y0;
                const f32* row0 =
                    texels.Data() + source.offset + static_cast<usize>(y0) * source.width;
                const f32* row1 =
                    texels.Data() + source.offset + static_cast<usize>(y1) * source.width;
                const f32 farthest = Min(Min(row0[x0], row0[x1]), Min(row1[x0], row1[x1]));
                texels[destination.offset + static_cast<usize>(y) * destination.width + x] =
                    farthest;
            }
        }
    }
    m_texels = static_cast<cyber::memory::DynamicArray<f32>&&>(texels);
    for (u32 level = 0; level < kHiZMaxLevels; ++level)
    {
        m_levels[level] = level < levelCount ? levels[level] : HiZLevel{};
    }
    m_levelCount = levelCount;
    return Status<Error>::Ok();
}

bool HiZPyramid::Empty() const noexcept
{
    return m_levelCount == 0;
}

u32 HiZPyramid::Width() const noexcept
{
    return m_levels[0].width;
}

u32 HiZPyramid::Height() const noexcept
{
    return m_levels[0].height;
}

u32 HiZPyramid::LevelCount() const noexcept
{
    return m_levelCount;
}

HiZLevel HiZPyramid::Level(u32 index) const noexcept
{
    return index < m_levelCount ? m_levels[index] : HiZLevel{};
}

Span<const f32> HiZPyramid::Texels() const noexcept
{
    return Span<const f32>(m_texels.Data(), m_texels.Size());
}

f32 HiZPyramid::Load(u32 level, u32 x, u32 y) const noexcept
{
    if (level >= m_levelCount)
    {
        return 0.0f;
    }
    const HiZLevel& shape = m_levels[level];
    const u32 cx = x < shape.width ? x : shape.width - 1;
    const u32 cy = y < shape.height ? y : shape.height - 1;
    return m_texels[shape.offset + static_cast<usize>(cy) * shape.width + cx];
}

HiZTestResult TestHiZOcclusion(const HiZPyramid& pyramid, const Mat4& viewProjection,
                               const Aabb& bounds) noexcept
{
    HiZTestResult result{};
    if (pyramid.Empty() || bounds.IsEmpty())
    {
        result.visibility = HiZVisibility::Untestable;
        result.depthMargin = kLargeMargin;
        result.pixelMargin = kLargeMargin;
        result.nearMargin = kLargeMargin;
        return result;
    }
    const Vec3 minimum = bounds.Min();
    const Vec3 maximum = bounds.Max();
    f32 minX = std::numeric_limits<f32>::max();
    f32 minY = std::numeric_limits<f32>::max();
    f32 maxX = -std::numeric_limits<f32>::max();
    f32 maxY = -std::numeric_limits<f32>::max();
    f32 maxZ = -std::numeric_limits<f32>::max();
    f32 minW = std::numeric_limits<f32>::max();
    for (u32 corner = 0; corner < 8; ++corner)
    {
        const Vec4 point{(corner & 1U) != 0 ? maximum.x : minimum.x,
                         (corner & 2U) != 0 ? maximum.y : minimum.y,
                         (corner & 4U) != 0 ? maximum.z : minimum.z, 1.0f};
        // `Mat4 * Vec4` sums the columns left to right; the shader does the
        // same so both round identically.
        const Vec4 clip = viewProjection * point;
        minW = Min(minW, clip.w);
        if (clip.w <= kHiZNearGuard)
        {
            continue;
        }
        const f32 x = clip.x / clip.w;
        const f32 y = clip.y / clip.w;
        const f32 z = clip.z / clip.w;
        minX = Min(minX, x);
        maxX = Max(maxX, x);
        minY = Min(minY, y);
        maxY = Max(maxY, y);
        maxZ = Max(maxZ, z);
    }
    result.depthMargin = kLargeMargin;
    result.pixelMargin = kLargeMargin;
    result.nearMargin = minW - kHiZNearGuard;
    if (minW <= kHiZNearGuard)
    {
        result.visibility = HiZVisibility::Untestable;
        return result;
    }
    if (maxX < -1.0f || minX > 1.0f || maxY < -1.0f || minY > 1.0f)
    {
        result.visibility = HiZVisibility::Untestable;
        // Half the view spans one NDC unit, so this is in pixels.
        const f32 outsideX =
            Max(-1.0f - maxX, minX - 1.0f) * 0.5f * static_cast<f32>(pyramid.Width());
        const f32 outsideY =
            Max(-1.0f - maxY, minY - 1.0f) * 0.5f * static_cast<f32>(pyramid.Height());
        result.pixelMargin = Max(outsideX, outsideY);
        return result;
    }
    minX = Max(minX, -1.0f);
    maxX = Min(maxX, 1.0f);
    minY = Max(minY, -1.0f);
    maxY = Min(maxY, 1.0f);

    const f32 width = static_cast<f32>(pyramid.Width());
    const f32 height = static_cast<f32>(pyramid.Height());
    // Texture space: +Y down, the top row first.
    const f32 left = (minX * 0.5f + 0.5f) * width;
    const f32 right = (maxX * 0.5f + 0.5f) * width;
    const f32 top = (0.5f - maxY * 0.5f) * height;
    const f32 bottom = (0.5f - minY * 0.5f) * height;
    const auto clampTexel = [](f32 value, u32 extent) noexcept
    {
        const f32 floored = std::floor(value);
        if (floored <= 0.0f)
        {
            return u32{0};
        }
        const u32 texel = static_cast<u32>(floored);
        return texel < extent ? texel : extent - 1;
    };
    const u32 x0 = clampTexel(left, pyramid.Width());
    const u32 x1 = clampTexel(right, pyramid.Width());
    const u32 y0 = clampTexel(top, pyramid.Height());
    const u32 y1 = clampTexel(bottom, pyramid.Height());

    // The finest level at which the rectangle spans at most two texels per
    // axis, so four reads cover it. Stepping the coordinates one level at a
    // time is the shader's form of `x >> level`.
    u32 level = 0;
    u32 lx0 = x0;
    u32 lx1 = x1;
    u32 ly0 = y0;
    u32 ly1 = y1;
    while (level + 1 < pyramid.LevelCount() && (lx1 - lx0 > 1 || ly1 - ly0 > 1))
    {
        lx0 >>= 1;
        lx1 >>= 1;
        ly0 >>= 1;
        ly1 >>= 1;
        ++level;
    }
    const f32 occluder = Min(Min(pyramid.Load(level, lx0, ly0), pyramid.Load(level, lx1, ly0)),
                             Min(pyramid.Load(level, lx0, ly1), pyramid.Load(level, lx1, ly1)));

    result.level = level;
    result.boxDepth = maxZ;
    result.occluderDepth = occluder;
    result.visibility = maxZ < occluder ? HiZVisibility::Occluded : HiZVisibility::Visible;
    result.depthMargin = occluder > 0.0f ? Abs(maxZ - occluder) / occluder : kLargeMargin;
    result.pixelMargin =
        Min(Min(FloorMargin(left), FloorMargin(right)), Min(FloorMargin(top), FloorMargin(bottom)));
    return result;
}

// ---------------------------------------------------------------------------
// View and decisions
// ---------------------------------------------------------------------------

GpuCullView MakeGpuCullView(const Mat4& viewProjection, Vec3 cameraPosition, f32 verticalFovRadians,
                            u32 viewportHeight, f32 nearDistance) noexcept
{
    GpuCullView view{};
    view.frustum = cyber::math::Frustum::FromViewProjection(viewProjection);
    view.cameraPosition = cameraPosition;
    view.projectionScalePixels =
        static_cast<f32>(viewportHeight) / (2.0f * std::tan(verticalFovRadians * 0.5f));
    view.nearDistance = nearDistance;
    view.occlusionViewProjection = viewProjection;
    view.flags = kGpuCullFrustum | kGpuCullLodSelection;
    return view;
}

GpuInstanceDecision DecideGpuInstance(const GpuCullInstance& instance, const GpuDrawGroup& group,
                                      const GpuCullView& view, const HiZPyramid* occluders) noexcept
{
    GpuInstanceDecision decision{};
    decision.frustumMargin = kLargeMargin;
    decision.lodMargin = kLargeMargin;
    const Vec3 minimum{instance.boundsMin[0], instance.boundsMin[1], instance.boundsMin[2]};
    const Vec3 maximum{instance.boundsMax[0], instance.boundsMax[1], instance.boundsMax[2]};

    if ((view.flags & kGpuCullFrustum) != 0)
    {
        // Phase 22's `Intersects`: the positive vertex of each plane, in the
        // frustum's plane order, rejected at the first negative distance.
        for (const cyber::math::Plane& plane : view.frustum.planes)
        {
            const Vec3 positive{plane.normal.x >= 0.0f ? maximum.x : minimum.x,
                                plane.normal.y >= 0.0f ? maximum.y : minimum.y,
                                plane.normal.z >= 0.0f ? maximum.z : minimum.z};
            const f32 distance = cyber::math::DistanceTo(plane, positive);
            if (distance < 0.0f)
            {
                decision.frustumCulled = true;
                decision.frustumMargin = -distance;
                return decision;
            }
            decision.frustumMargin = Min(decision.frustumMargin, distance);
        }
    }

    if ((view.flags & kGpuCullLodSelection) != 0 && group.lodCount > 1)
    {
        // Phase 22's screen-space LOD rule, operation for operation.
        const Vec3 center = (minimum + maximum) * 0.5f;
        const Vec3 extents = (maximum - minimum) * 0.5f;
        const f32 radius = cyber::math::Length(extents);
        const f32 distance = cyber::math::Length(center - view.cameraPosition);
        const f32 denominator = Max(distance - radius, view.nearDistance);
        const f32 projectedRadius = radius * view.projectionScalePixels / denominator;
        for (u32 lod = 0; lod + 1 < group.lodCount; ++lod)
        {
            const f32 threshold = group.lodPixelThresholds[lod];
            if (projectedRadius < threshold)
            {
                decision.lod = lod + 1;
            }
            decision.lodMargin =
                Min(decision.lodMargin, Abs(projectedRadius - threshold) / threshold);
        }
    }

    if ((view.flags & kGpuCullOcclusion) != 0 && occluders != nullptr)
    {
        decision.occlusion = TestHiZOcclusion(*occluders, view.occlusionViewProjection,
                                              Aabb::FromMinMax(minimum, maximum));
        if (decision.occlusion.visibility == HiZVisibility::Occluded)
        {
            decision.occlusionCulled = true;
            return decision;
        }
    }

    decision.visible = true;
    decision.batch = group.firstBatch + decision.lod;
    return decision;
}

GpuCullOutput::GpuCullOutput(cyber::memory::Allocator& allocator) noexcept
    : allocator(&allocator), batchCounts(allocator), visibleInstances(allocator), draws(allocator)
{
}

namespace
{

// One decision per instance, written by the parallel pass and consumed by the
// serial append, which is what makes the result independent of threading.
struct PackedDecision
{
    u32 batch = 0;
    u8 visible = 0;
    u8 frustumCulled = 0;
    u8 occlusionCulled = 0;
    u8 reserved = 0;
};

inline constexpr u32 kReferenceChunk = 4096;

struct ReferenceContext
{
    const GpuDrivenScene* scene = nullptr;
    const GpuCullView* view = nullptr;
    const HiZPyramid* occluders = nullptr;
    PackedDecision* decisions = nullptr;
    u32 instanceCount = 0;
};

void DecideChunk(void* userData, u32 chunk) noexcept
{
    auto& context = *static_cast<ReferenceContext*>(userData);
    const Span<const GpuCullInstance> instances = context.scene->Instances();
    const Span<const GpuDrawGroup> groups = context.scene->Groups();
    const u32 begin = chunk * kReferenceChunk;
    const u32 end = begin + kReferenceChunk < context.instanceCount ? begin + kReferenceChunk
                                                                    : context.instanceCount;
    for (u32 index = begin; index < end; ++index)
    {
        const GpuCullInstance& instance = instances[index];
        const GpuInstanceDecision decision = DecideGpuInstance(instance, groups[instance.drawGroup],
                                                               *context.view, context.occluders);
        PackedDecision& packed = context.decisions[index];
        packed.batch = decision.batch;
        packed.visible = decision.visible ? 1 : 0;
        packed.frustumCulled = decision.frustumCulled ? 1 : 0;
        packed.occlusionCulled = decision.occlusionCulled ? 1 : 0;
    }
}

}  // namespace

Status<Error> CullInstancesReference(const GpuDrivenScene& scene, const GpuCullView& view,
                                     const HiZPyramid* occluders, GpuCullOutput& output,
                                     cyber::jobs::JobSystem* jobs) noexcept
{
    if (!scene.IsFinalized())
    {
        return Fail(ErrorCode::InvalidArgument, "gpu-driven reference: the scene is not finalized");
    }
    if ((view.flags & kGpuCullOcclusion) != 0 && (occluders == nullptr || occluders->Empty()))
    {
        return Fail(ErrorCode::InvalidArgument,
                    "gpu-driven reference: occlusion culling needs a pyramid");
    }
    const Span<const GpuCullInstance> instances = scene.Instances();
    const Span<const GpuDrawBatch> batches = scene.Batches();
    const u32 instanceCount = static_cast<u32>(instances.Size());

    cyber::memory::DynamicArray<PackedDecision> decisions(*output.allocator);
    if (!decisions.TryResize(instanceCount) || !output.batchCounts.TryResize(batches.Size()) ||
        !output.visibleInstances.TryResize(scene.VisibleCapacity()) ||
        !output.draws.TryReserve(batches.Size()))
    {
        return Fail(ErrorCode::Internal, "gpu-driven reference: allocation failed");
    }

    ReferenceContext context{};
    context.scene = &scene;
    context.view = &view;
    context.occluders = (view.flags & kGpuCullOcclusion) != 0 ? occluders : nullptr;
    context.decisions = decisions.Data();
    context.instanceCount = instanceCount;
    const u32 chunks = (instanceCount + kReferenceChunk - 1) / kReferenceChunk;
    if (jobs != nullptr && chunks > 1)
    {
        jobs->ParallelFor(chunks, 1, &DecideChunk, &context);
    }
    else
    {
        for (u32 chunk = 0; chunk < chunks; ++chunk)
        {
            DecideChunk(&context, chunk);
        }
    }

    for (usize batch = 0; batch < batches.Size(); ++batch)
    {
        output.batchCounts[batch] = 0;
    }
    for (usize index = 0; index < output.visibleInstances.Size(); ++index)
    {
        output.visibleInstances[index] = 0;
    }
    output.draws.Clear();
    GpuCullCounters counters{};
    for (u32 index = 0; index < instanceCount; ++index)
    {
        const PackedDecision& decision = decisions[index];
        counters.frustumCulled += decision.frustumCulled;
        counters.occlusionCulled += decision.occlusionCulled;
        if (decision.visible == 0)
        {
            continue;
        }
        ++counters.visible;
        const GpuDrawBatch& batch = batches[decision.batch];
        u32& slot = output.batchCounts[decision.batch];
        if (slot >= batch.capacity)
        {
            ++counters.overflow;
            continue;
        }
        output.visibleInstances[batch.visibleBase + slot] = instances[index].instanceId;
        ++slot;
    }
    for (usize index = 0; index < batches.Size(); ++index)
    {
        const u32 count = output.batchCounts[index];
        if (count == 0)
        {
            continue;
        }
        const GpuDrawBatch& batch = batches[index];
        cyber::rhi::DrawIndexedArguments draw{};
        draw.indexCount = batch.indexCount;
        draw.instanceCount = count;
        draw.firstIndex = batch.firstIndex;
        draw.vertexOffset = batch.vertexOffset;
        draw.firstInstance = batch.visibleBase;
        (void)output.draws.TryPushBack(draw);
    }
    counters.drawCount = static_cast<u32>(output.draws.Size());
    output.counters = counters;
    return Status<Error>::Ok();
}

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

GeometryPathPlan SelectGeometryPath(const cyber::rhi::Capabilities& capabilities,
                                    GeometryPathPreference preference) noexcept
{
    GeometryPathPlan plan{};
    plan.meshShadersReported = capabilities.features.Has(cyber::rhi::Feature::MeshShaders);
    plan.path = GeometryPath::VertexPipeline;
    if (preference == GeometryPathPreference::VertexOnly)
    {
        plan.reason = GeometryFallbackReason::PolicyDisabled;
    }
    else if (!plan.meshShadersReported)
    {
        plan.reason = GeometryFallbackReason::Unsupported;
    }
    else if (!kMeshShaderPathCompiledIn)
    {
        plan.reason = GeometryFallbackReason::NotImplemented;
    }
    else
    {
        plan.path = GeometryPath::MeshShader;
        plan.reason = GeometryFallbackReason::None;
    }
    return plan;
}

const char* GeometryPathName(GeometryPath path) noexcept
{
    switch (path)
    {
    case GeometryPath::VertexPipeline:
        return "vertex_pipeline";
    case GeometryPath::MeshShader:
        return "mesh_shader";
    }
    return "unknown";
}

const char* GeometryFallbackReasonName(GeometryFallbackReason reason) noexcept
{
    switch (reason)
    {
    case GeometryFallbackReason::None:
        return "none";
    case GeometryFallbackReason::PolicyDisabled:
        return "policy_disabled";
    case GeometryFallbackReason::Unsupported:
        return "unsupported";
    case GeometryFallbackReason::NotImplemented:
        return "not_implemented";
    }
    return "unknown";
}

AsyncComputePlan SelectAsyncComputePlan(const AsyncComputePolicyDesc& policy,
                                        bool computeQueueAvailable,
                                        const AsyncComputeMeasurement* measurement) noexcept
{
    AsyncComputePlan plan{};
    if (measurement != nullptr && measurement->serialMilliseconds > 0.0f &&
        IsFinite(measurement->serialMilliseconds) && IsFinite(measurement->asyncMilliseconds))
    {
        plan.measuredGainFraction =
            (measurement->serialMilliseconds - measurement->asyncMilliseconds) /
            measurement->serialMilliseconds;
    }
    const auto serial = [&plan](AsyncComputeReason reason) noexcept
    {
        plan.useAsyncCompute = false;
        plan.cullingQueue = cyber::rhi::QueueKind::Graphics;
        plan.lightCullingQueue = cyber::rhi::QueueKind::Graphics;
        plan.reason = reason;
        return plan;
    };
    if (policy.preference == AsyncComputePreference::Off)
    {
        return serial(AsyncComputeReason::PolicyDisabled);
    }
    if (!computeQueueAvailable)
    {
        return serial(AsyncComputeReason::NoComputeQueue);
    }
    if (policy.preference == AsyncComputePreference::Auto)
    {
        if (measurement == nullptr || measurement->samples < policy.minimumSamples ||
            !(measurement->serialMilliseconds > 0.0f) ||
            !IsFinite(measurement->asyncMilliseconds) || measurement->asyncMilliseconds < 0.0f)
        {
            return serial(AsyncComputeReason::NotMeasured);
        }
        if (plan.measuredGainFraction < policy.minimumGainFraction)
        {
            return serial(AsyncComputeReason::NoMeasuredBenefit);
        }
    }
    plan.useAsyncCompute = true;
    plan.cullingQueue = cyber::rhi::QueueKind::Compute;
    plan.lightCullingQueue = cyber::rhi::QueueKind::Compute;
    plan.reason = AsyncComputeReason::None;
    return plan;
}

const char* AsyncComputeReasonName(AsyncComputeReason reason) noexcept
{
    switch (reason)
    {
    case AsyncComputeReason::None:
        return "none";
    case AsyncComputeReason::PolicyDisabled:
        return "policy_disabled";
    case AsyncComputeReason::NoComputeQueue:
        return "no_compute_queue";
    case AsyncComputeReason::NotMeasured:
        return "not_measured";
    case AsyncComputeReason::NoMeasuredBenefit:
        return "no_measured_benefit";
    }
    return "unknown";
}

}  // namespace cyber::renderer
