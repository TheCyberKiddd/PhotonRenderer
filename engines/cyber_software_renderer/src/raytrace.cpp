#include <cyber/software/raytrace.h>

#include <algorithm>
#include <cmath>
#include <new>
#include <utility>

namespace cyber::software
{

namespace
{

using cyber::core::Error;
using cyber::core::ErrorCode;

// Leaves the surface-area heuristic may keep without being forced to split,
// whatever maxLeafTriangles says. Above this a split is always taken, so a
// pathological cost function cannot produce one enormous leaf.
constexpr u32 kSahLeafCap = 32;
constexpr u32 kMaxBinCount = 64;
constexpr u32 kTraversalStackSize = kMaxBvhDepth + 2;

usize AlignUp(usize value, usize alignment) noexcept
{
    return (value + alignment - 1) / alignment * alignment;
}

bool IsFinite(const math::Vec3& value) noexcept
{
    return math::IsFinite(value.x) && math::IsFinite(value.y) && math::IsFinite(value.z);
}

f32 Component(const math::Vec3& value, u32 axis) noexcept
{
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

f32 SurfaceArea(const math::Vec3& minimum, const math::Vec3& maximum) noexcept
{
    const math::Vec3 extent = maximum - minimum;
    if (extent.x < 0.0f || extent.y < 0.0f || extent.z < 0.0f)
    {
        return 0.0f;
    }
    return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
}

// Bounds and centroid of one primitive, which is all the builder needs to know
// about it: the same builder orders triangles and instances.
struct PrimitiveRef
{
    math::Vec3 minimum{};
    math::Vec3 maximum{};
    math::Vec3 centroid{};
};

struct BinBounds
{
    math::Vec3 minimum{kRayMaxDistance, kRayMaxDistance, kRayMaxDistance};
    math::Vec3 maximum{-kRayMaxDistance, -kRayMaxDistance, -kRayMaxDistance};
    u32 count = 0;

    void Grow(const PrimitiveRef& primitive) noexcept
    {
        minimum = math::Min(minimum, primitive.minimum);
        maximum = math::Max(maximum, primitive.maximum);
        ++count;
    }

    void Grow(const BinBounds& other) noexcept
    {
        if (other.count == 0)
        {
            return;
        }
        minimum = math::Min(minimum, other.minimum);
        maximum = math::Max(maximum, other.maximum);
        count += other.count;
    }

    [[nodiscard]] f32 Area() const noexcept
    {
        return count == 0 ? 0.0f : SurfaceArea(minimum, maximum);
    }
};

// A node is a box plus either two children, stored adjacently so one index
// names both, or a contiguous run of primitives. count == 0 marks an interior
// node, so the layout is 32 bytes with no flag field.
template <typename NodeT>
struct BvhBuilder
{
    const PrimitiveRef* primitives = nullptr;
    u32* order = nullptr;
    NodeT* nodes = nullptr;
    u32 nodeCount = 0;
    u32 leafCount = 0;
    u32 maxDepth = 0;
    u32 largestLeaf = 0;
    u32 maxLeafSize = 1;
    u32 binCount = 16;

    void MakeLeaf(NodeT& node, u32 first, u32 count, u32 depth) noexcept
    {
        node.leftOrFirst = first;
        node.count = count;
        ++leafCount;
        maxDepth = std::max(maxDepth, depth);
        largestLeaf = std::max(largestLeaf, count);
    }

    // Splits at the median centroid along the widest centroid axis. This is the
    // fallback for when binning cannot separate the primitives -- every centroid
    // in one bin -- and it always produces two non-empty halves. Ties are broken
    // by primitive index so the order is fully determined by the input.
    u32 MedianSplit(u32 first, u32 count, u32 axis) noexcept
    {
        const u32 half = count / 2;
        const PrimitiveRef* refs = primitives;
        std::nth_element(order + first, order + first + half, order + first + count,
                         [refs, axis](u32 lhs, u32 rhs) noexcept
                         {
                             const f32 a = Component(refs[lhs].centroid, axis);
                             const f32 b = Component(refs[rhs].centroid, axis);
                             return a < b || (a == b && lhs < rhs);
                         });
        return half;
    }

    void Build(u32 nodeIndex, u32 first, u32 count, u32 depth) noexcept
    {
        NodeT& node = nodes[nodeIndex];
        BinBounds bounds;
        math::Vec3 centroidMin{kRayMaxDistance, kRayMaxDistance, kRayMaxDistance};
        math::Vec3 centroidMax{-kRayMaxDistance, -kRayMaxDistance, -kRayMaxDistance};
        for (u32 index = first; index < first + count; ++index)
        {
            const PrimitiveRef& primitive = primitives[order[index]];
            bounds.Grow(primitive);
            centroidMin = math::Min(centroidMin, primitive.centroid);
            centroidMax = math::Max(centroidMax, primitive.centroid);
        }
        node.boundsMin = bounds.minimum;
        node.boundsMax = bounds.maximum;

        // The depth limit is a guarantee to the traversal, not a heuristic: a
        // node at the limit is a leaf whatever its size.
        if (count <= maxLeafSize || depth + 1 >= kMaxBvhDepth)
        {
            MakeLeaf(node, first, count, depth);
            return;
        }

        const f32 parentArea = bounds.Area();
        f32 bestCost = kRayMaxDistance;
        u32 bestAxis = 0;
        u32 bestBoundary = 0;
        bool haveSplit = false;
        for (u32 axis = 0; axis < 3; ++axis)
        {
            const f32 low = Component(centroidMin, axis);
            const f32 extent = Component(centroidMax, axis) - low;
            if (!(extent > 0.0f))
            {
                continue;
            }
            BinBounds bins[kMaxBinCount];
            const f32 scale = static_cast<f32>(binCount) / extent;
            for (u32 index = first; index < first + count; ++index)
            {
                const PrimitiveRef& primitive = primitives[order[index]];
                const f32 position = (Component(primitive.centroid, axis) - low) * scale;
                const u32 bin = std::min(binCount - 1, static_cast<u32>(std::max(0.0f, position)));
                bins[bin].Grow(primitive);
            }

            f32 leftArea[kMaxBinCount] = {};
            u32 leftCount[kMaxBinCount] = {};
            BinBounds running;
            for (u32 bin = 0; bin + 1 < binCount; ++bin)
            {
                running.Grow(bins[bin]);
                leftArea[bin] = running.Area();
                leftCount[bin] = running.count;
            }
            running = BinBounds{};
            for (u32 bin = binCount - 1; bin > 0; --bin)
            {
                running.Grow(bins[bin]);
                const u32 boundary = bin - 1;
                if (leftCount[boundary] == 0 || running.count == 0)
                {
                    continue;
                }
                const f32 cost = leftArea[boundary] * static_cast<f32>(leftCount[boundary]) +
                                 running.Area() * static_cast<f32>(running.count);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestAxis = axis;
                    bestBoundary = boundary;
                    haveSplit = true;
                }
            }
        }

        // Traversal and intersection are costed equally. A split that is no
        // cheaper than testing every primitive here is declined, as long as the
        // leaf it leaves behind is a reasonable size.
        const f32 leafCost = static_cast<f32>(count);
        const f32 splitCost =
            haveSplit && parentArea > 0.0f ? 1.0f + bestCost / parentArea : kRayMaxDistance;
        if (splitCost >= leafCost && count <= kSahLeafCap)
        {
            MakeLeaf(node, first, count, depth);
            return;
        }

        u32 leftSize = 0;
        if (haveSplit)
        {
            const f32 low = Component(centroidMin, bestAxis);
            const f32 extent = Component(centroidMax, bestAxis) - low;
            const f32 scale = static_cast<f32>(binCount) / extent;
            const PrimitiveRef* refs = primitives;
            const u32 bins = binCount;
            u32* middle = std::partition(
                order + first, order + first + count,
                [refs, bestAxis, low, scale, bins, bestBoundary](u32 index) noexcept
                {
                    const f32 position = (Component(refs[index].centroid, bestAxis) - low) * scale;
                    const u32 bin = std::min(bins - 1, static_cast<u32>(std::max(0.0f, position)));
                    return bin <= bestBoundary;
                });
            leftSize = static_cast<u32>(middle - (order + first));
        }
        if (leftSize == 0 || leftSize == count)
        {
            const math::Vec3 extent = centroidMax - centroidMin;
            const u32 axis = extent.x >= extent.y && extent.x >= extent.z
                                 ? 0u
                                 : (extent.y >= extent.z ? 1u : 2u);
            leftSize = MedianSplit(first, count, axis);
        }

        const u32 left = nodeCount;
        nodeCount += 2;
        node.leftOrFirst = left;
        node.count = 0;
        Build(left, first, leftSize, depth + 1);
        Build(left + 1, first + leftSize, count - leftSize, depth + 1);
    }
};

// Slab test against a node box. invDirection has no zero component (the caller
// clamps them), so no 0 * inf NaN can appear. Returns the entry distance, or a
// value above tMax for a miss.
template <typename NodeT>
f32 IntersectNode(const NodeT& node, const math::Vec3& origin, const math::Vec3& invDirection,
                  f32 tMin, f32 tMax) noexcept
{
    const f32 tx1 = (node.boundsMin.x - origin.x) * invDirection.x;
    const f32 tx2 = (node.boundsMax.x - origin.x) * invDirection.x;
    f32 enter = std::min(tx1, tx2);
    f32 exit = std::max(tx1, tx2);
    const f32 ty1 = (node.boundsMin.y - origin.y) * invDirection.y;
    const f32 ty2 = (node.boundsMax.y - origin.y) * invDirection.y;
    enter = std::max(enter, std::min(ty1, ty2));
    exit = std::min(exit, std::max(ty1, ty2));
    const f32 tz1 = (node.boundsMin.z - origin.z) * invDirection.z;
    const f32 tz2 = (node.boundsMax.z - origin.z) * invDirection.z;
    enter = std::max(enter, std::min(tz1, tz2));
    exit = std::min(exit, std::max(tz1, tz2));
    enter = std::max(enter, tMin);
    exit = std::min(exit, tMax);
    return enter <= exit ? enter : kRayMaxDistance * 2.0f;
}

math::Vec3 SafeInverse(const math::Vec3& direction) noexcept
{
    constexpr f32 kTiny = 1.0e-20f;
    const auto invert = [](f32 value) noexcept
    {
        if (std::fabs(value) < kTiny)
        {
            value = value < 0.0f ? -kTiny : kTiny;
        }
        return 1.0f / value;
    };
    return math::Vec3{invert(direction.x), invert(direction.y), invert(direction.z)};
}

bool ValidRay(const RayDesc& ray) noexcept
{
    return IsFinite(ray.origin) && IsFinite(ray.direction) &&
           math::LengthSquared(ray.direction) > 0.0f && math::IsFinite(ray.tMin) &&
           !std::isnan(ray.tMax) && ray.tMin >= 0.0f && ray.tMin <= ray.tMax;
}

math::Vec3 TransformBoxCorner(const math::Mat4& transform, const math::Vec3& minimum,
                              const math::Vec3& maximum, u32 corner) noexcept
{
    const math::Vec3 point{(corner & 1u) != 0 ? maximum.x : minimum.x,
                           (corner & 2u) != 0 ? maximum.y : minimum.y,
                           (corner & 4u) != 0 ? maximum.z : minimum.z};
    return transform.TransformPoint(point);
}

bool IsFinite(const math::Mat4& matrix) noexcept
{
    for (usize column = 0; column < 4; ++column)
    {
        const math::Vec4& value = matrix[column];
        if (!math::IsFinite(value.x) || !math::IsFinite(value.y) || !math::IsFinite(value.z) ||
            !math::IsFinite(value.w))
        {
            return false;
        }
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Bottom level
// ---------------------------------------------------------------------------

struct BottomLevelAccelerationStructure::Triangle
{
    math::Vec3 p0{};
    math::Vec3 edge1{};
    math::Vec3 edge2{};
    math::Vec3 n0{};
    math::Vec3 n1{};
    math::Vec3 n2{};
    math::Vec2 uv0{};
    math::Vec2 uv1{};
    math::Vec2 uv2{};
    u32 primitiveIndex = 0;
};

struct BottomLevelAccelerationStructure::Node
{
    math::Vec3 boundsMin{};
    u32 leftOrFirst = 0;
    math::Vec3 boundsMax{};
    u32 count = 0;
};

namespace
{

// Möller-Trumbore without back-face culling. The determinant test is relative
// to the edge lengths, so a small triangle is not rejected as degenerate just
// because it is small.
template <typename TriangleT>
bool IntersectTriangle(const TriangleT& triangle, const math::Vec3& origin,
                       const math::Vec3& direction, f32 tMin, f32 tMax, f32& distance, f32& u,
                       f32& v) noexcept
{
    const math::Vec3 p = math::Cross(direction, triangle.edge2);
    const f32 determinant = math::Dot(triangle.edge1, p);
    const f32 scale =
        math::Length(triangle.edge1) * math::Length(triangle.edge2) * math::Length(direction);
    if (!(std::fabs(determinant) > 1.0e-12f * scale) || scale <= 0.0f)
    {
        return false;
    }
    const f32 inverse = 1.0f / determinant;
    const math::Vec3 offset = origin - triangle.p0;
    u = math::Dot(offset, p) * inverse;
    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }
    const math::Vec3 q = math::Cross(offset, triangle.edge1);
    v = math::Dot(direction, q) * inverse;
    if (v < 0.0f || u + v > 1.0f)
    {
        return false;
    }
    distance = math::Dot(triangle.edge2, q) * inverse;
    return distance >= tMin && distance <= tMax;
}

}  // namespace

BottomLevelAccelerationStructure::~BottomLevelAccelerationStructure()
{
    Release();
}

BottomLevelAccelerationStructure::BottomLevelAccelerationStructure(
    BottomLevelAccelerationStructure&& other) noexcept
{
    MoveFrom(other);
}

BottomLevelAccelerationStructure&
BottomLevelAccelerationStructure::operator=(BottomLevelAccelerationStructure&& other) noexcept
{
    if (this != &other)
    {
        Release();
        MoveFrom(other);
    }
    return *this;
}

void BottomLevelAccelerationStructure::Release() noexcept
{
    if (m_allocator != nullptr && m_storage != nullptr)
    {
        m_allocator->Deallocate(m_storage);
    }
    m_allocator = nullptr;
    m_storage = nullptr;
    m_storageBytes = 0;
    m_triangles = nullptr;
    m_nodes = nullptr;
    m_triangleCount = 0;
    m_nodeCount = 0;
    m_leafCount = 0;
    m_maxDepth = 0;
    m_maxLeafTriangles = 0;
}

void BottomLevelAccelerationStructure::MoveFrom(BottomLevelAccelerationStructure& other) noexcept
{
    m_allocator = std::exchange(other.m_allocator, nullptr);
    m_storage = std::exchange(other.m_storage, nullptr);
    m_storageBytes = std::exchange(other.m_storageBytes, 0);
    m_triangles = std::exchange(other.m_triangles, nullptr);
    m_nodes = std::exchange(other.m_nodes, nullptr);
    m_triangleCount = std::exchange(other.m_triangleCount, 0);
    m_nodeCount = std::exchange(other.m_nodeCount, 0);
    m_leafCount = std::exchange(other.m_leafCount, 0);
    m_maxDepth = std::exchange(other.m_maxDepth, 0);
    m_maxLeafTriangles = std::exchange(other.m_maxLeafTriangles, 0);
}

core::Result<BottomLevelAccelerationStructure, core::Error>
BottomLevelAccelerationStructure::Build(const AccelerationStructureBuildDesc& desc,
                                        memory::Allocator& allocator) noexcept
{
    using ResultType = core::Result<BottomLevelAccelerationStructure, core::Error>;
    const usize vertexCount = desc.vertices.Size();
    if (vertexCount == 0 || desc.vertices.Data() == nullptr)
    {
        return ResultType::Err(ErrorCode::InvalidArgument,
                               "acceleration structure has no geometry");
    }
    if (vertexCount % 3 != 0)
    {
        return ResultType::Err(ErrorCode::InvalidArgument,
                               "acceleration structure vertex count is not a multiple of three");
    }
    if (vertexCount / 3 > kMaxBlasTriangles)
    {
        return ResultType::Err(ErrorCode::InvalidArgument,
                               "acceleration structure has too many triangles");
    }
    if (desc.maxLeafTriangles < 1 || desc.maxLeafTriangles > kSahLeafCap || desc.binCount < 2 ||
        desc.binCount > kMaxBinCount)
    {
        return ResultType::Err(ErrorCode::InvalidArgument,
                               "acceleration structure build descriptor out of range");
    }
    const VertexInput* vertices = desc.vertices.Data();
    for (usize index = 0; index < vertexCount; ++index)
    {
        if (!IsFinite(vertices[index].position))
        {
            return ResultType::Err(ErrorCode::InvalidArgument,
                                   "acceleration structure has a non-finite vertex");
        }
    }

    const u32 triangleCount = static_cast<u32>(vertexCount / 3);
    const u32 nodeCapacity = 2 * triangleCount - 1;
    const usize triangleBytes = AlignUp(sizeof(Triangle) * triangleCount, 16);
    const usize nodeBytes = AlignUp(sizeof(Node) * nodeCapacity, 16);
    const usize storageBytes = triangleBytes + nodeBytes;

    // Build scratch: primitive bounds and the permutation the builder sorts.
    const usize refBytes = AlignUp(sizeof(PrimitiveRef) * triangleCount, 16);
    const usize orderBytes = AlignUp(sizeof(u32) * triangleCount, 16);
    void* scratch = allocator.Allocate(refBytes + orderBytes, 16);
    if (scratch == nullptr)
    {
        return ResultType::Err(ErrorCode::Internal,
                               "acceleration structure scratch allocation failed");
    }
    void* storage = allocator.Allocate(storageBytes, 16);
    if (storage == nullptr)
    {
        allocator.Deallocate(scratch);
        return ResultType::Err(ErrorCode::Internal, "acceleration structure allocation failed");
    }

    auto* refs = static_cast<PrimitiveRef*>(scratch);
    auto* order = reinterpret_cast<u32*>(static_cast<unsigned char*>(scratch) + refBytes);
    for (u32 triangle = 0; triangle < triangleCount; ++triangle)
    {
        const math::Vec3& a = vertices[triangle * 3 + 0].position;
        const math::Vec3& b = vertices[triangle * 3 + 1].position;
        const math::Vec3& c = vertices[triangle * 3 + 2].position;
        PrimitiveRef& ref = refs[triangle];
        ref.minimum = math::Min(a, math::Min(b, c));
        ref.maximum = math::Max(a, math::Max(b, c));
        ref.centroid = (a + b + c) * (1.0f / 3.0f);
        order[triangle] = triangle;
    }

    BottomLevelAccelerationStructure result;
    result.m_allocator = &allocator;
    result.m_storage = storage;
    result.m_storageBytes = storageBytes;
    result.m_triangles = static_cast<Triangle*>(storage);
    result.m_nodes = reinterpret_cast<Node*>(static_cast<unsigned char*>(storage) + triangleBytes);
    for (u32 node = 0; node < nodeCapacity; ++node)
    {
        new (&result.m_nodes[node]) Node{};
    }

    BvhBuilder<Node> builder;
    builder.primitives = refs;
    builder.order = order;
    builder.nodes = result.m_nodes;
    builder.nodeCount = 1;
    builder.maxLeafSize = desc.maxLeafTriangles;
    builder.binCount = desc.binCount;
    builder.Build(0, 0, triangleCount, 0);

    // Store triangles in leaf order, so a leaf's triangles are contiguous and
    // the traversal reads them sequentially.
    for (u32 slot = 0; slot < triangleCount; ++slot)
    {
        const u32 source = order[slot];
        const VertexInput& a = vertices[source * 3 + 0];
        const VertexInput& b = vertices[source * 3 + 1];
        const VertexInput& c = vertices[source * 3 + 2];
        Triangle* triangle = new (&result.m_triangles[slot]) Triangle{};
        triangle->p0 = a.position;
        triangle->edge1 = b.position - a.position;
        triangle->edge2 = c.position - a.position;
        triangle->n0 = a.normal;
        triangle->n1 = b.normal;
        triangle->n2 = c.normal;
        triangle->uv0 = a.uv;
        triangle->uv1 = b.uv;
        triangle->uv2 = c.uv;
        triangle->primitiveIndex = source;
    }
    allocator.Deallocate(scratch);

    result.m_triangleCount = triangleCount;
    result.m_nodeCount = builder.nodeCount;
    result.m_leafCount = builder.leafCount;
    result.m_maxDepth = builder.maxDepth;
    result.m_maxLeafTriangles = builder.largestLeaf;
    return ResultType::Ok(std::move(result));
}

bool BottomLevelAccelerationStructure::IsValid() const noexcept
{
    return m_storage != nullptr && m_triangleCount > 0 && m_nodeCount > 0;
}

u32 BottomLevelAccelerationStructure::TriangleCount() const noexcept
{
    return m_triangleCount;
}

math::Aabb BottomLevelAccelerationStructure::Bounds() const noexcept
{
    if (!IsValid())
    {
        return math::Aabb::Empty();
    }
    return math::Aabb::FromMinMax(m_nodes[0].boundsMin, m_nodes[0].boundsMax);
}

AccelerationStructureStats BottomLevelAccelerationStructure::Stats() const noexcept
{
    AccelerationStructureStats stats{};
    stats.triangleCount = m_triangleCount;
    stats.nodeCount = m_nodeCount;
    stats.leafCount = m_leafCount;
    stats.maxDepth = m_maxDepth;
    stats.maxLeafTriangles = m_maxLeafTriangles;
    return stats;
}

usize BottomLevelAccelerationStructure::DataSizeBytes() const noexcept
{
    return m_storageBytes;
}

bool BottomLevelAccelerationStructure::Intersect(const RayDesc& ray, RayHit& out,
                                                 RayTraversalStats* stats) const noexcept
{
    if (!IsValid() || !ValidRay(ray))
    {
        return false;
    }
    const math::Vec3 invDirection = SafeInverse(ray.direction);
    RayTraversalStats local{};
    f32 closest = ray.tMax;
    const Triangle* best = nullptr;
    f32 bestU = 0.0f;
    f32 bestV = 0.0f;

    u32 stack[kTraversalStackSize];
    u32 stackSize = 0;
    u32 current = 0;
    ++local.nodeTests;
    if (IntersectNode(m_nodes[0], ray.origin, invDirection, ray.tMin, closest) > closest)
    {
        if (stats != nullptr)
        {
            stats->Add(local);
        }
        return false;
    }
    for (;;)
    {
        const Node& node = m_nodes[current];
        if (node.count > 0)
        {
            for (u32 index = node.leftOrFirst; index < node.leftOrFirst + node.count; ++index)
            {
                ++local.triangleTests;
                f32 distance = 0.0f;
                f32 u = 0.0f;
                f32 v = 0.0f;
                if (IntersectTriangle(m_triangles[index], ray.origin, ray.direction, ray.tMin,
                                      closest, distance, u, v))
                {
                    closest = distance;
                    best = &m_triangles[index];
                    bestU = u;
                    bestV = v;
                }
            }
        }
        else
        {
            const u32 left = node.leftOrFirst;
            const u32 right = left + 1;
            local.nodeTests += 2;
            f32 leftDistance =
                IntersectNode(m_nodes[left], ray.origin, invDirection, ray.tMin, closest);
            f32 rightDistance =
                IntersectNode(m_nodes[right], ray.origin, invDirection, ray.tMin, closest);
            u32 nearChild = left;
            u32 farChild = right;
            if (rightDistance < leftDistance)
            {
                std::swap(leftDistance, rightDistance);
                std::swap(nearChild, farChild);
            }
            if (leftDistance <= closest)
            {
                if (rightDistance <= closest && stackSize < kTraversalStackSize)
                {
                    stack[stackSize++] = farChild;
                }
                current = nearChild;
                continue;
            }
        }
        // Pop, skipping any node the current closest hit has made unreachable.
        bool found = false;
        while (stackSize > 0)
        {
            const u32 candidate = stack[--stackSize];
            ++local.nodeTests;
            if (IntersectNode(m_nodes[candidate], ray.origin, invDirection, ray.tMin, closest) <=
                closest)
            {
                current = candidate;
                found = true;
                break;
            }
        }
        if (!found)
        {
            break;
        }
    }

    if (stats != nullptr)
    {
        stats->Add(local);
    }
    if (best == nullptr)
    {
        return false;
    }

    const f32 w = 1.0f - bestU - bestV;
    math::Vec3 geometric = math::Normalize(math::Cross(best->edge1, best->edge2));
    const bool frontFace = math::Dot(geometric, ray.direction) < 0.0f;
    math::Vec3 shading = math::Normalize(best->n0 * w + best->n1 * bestU + best->n2 * bestV);
    if (!IsFinite(shading) || math::LengthSquared(shading) < 0.5f)
    {
        shading = geometric;
    }
    if (!frontFace)
    {
        geometric = -geometric;
        shading = -shading;
    }
    out.hit = true;
    out.distance = closest;
    out.position = ray.origin + ray.direction * closest;
    out.geometricNormal = geometric;
    out.shadingNormal = shading;
    out.uv = best->uv0 * w + best->uv1 * bestU + best->uv2 * bestV;
    out.barycentricU = bestU;
    out.barycentricV = bestV;
    out.instanceIndex = kInvalidRayIndex;
    out.instanceId = 0;
    out.primitiveIndex = best->primitiveIndex;
    out.frontFace = frontFace;
    return true;
}

bool BottomLevelAccelerationStructure::IntersectAny(const RayDesc& ray,
                                                    RayTraversalStats* stats) const noexcept
{
    if (!IsValid() || !ValidRay(ray))
    {
        return false;
    }
    const math::Vec3 invDirection = SafeInverse(ray.direction);
    RayTraversalStats local{};
    u32 stack[kTraversalStackSize];
    u32 stackSize = 0;
    stack[stackSize++] = 0;
    bool hit = false;
    while (stackSize > 0 && !hit)
    {
        const Node& node = m_nodes[stack[--stackSize]];
        ++local.nodeTests;
        if (IntersectNode(node, ray.origin, invDirection, ray.tMin, ray.tMax) > ray.tMax)
        {
            continue;
        }
        if (node.count > 0)
        {
            for (u32 index = node.leftOrFirst; index < node.leftOrFirst + node.count; ++index)
            {
                ++local.triangleTests;
                f32 distance = 0.0f;
                f32 u = 0.0f;
                f32 v = 0.0f;
                if (IntersectTriangle(m_triangles[index], ray.origin, ray.direction, ray.tMin,
                                      ray.tMax, distance, u, v))
                {
                    hit = true;
                    break;
                }
            }
        }
        else if (stackSize + 2 <= kTraversalStackSize)
        {
            stack[stackSize++] = node.leftOrFirst + 1;
            stack[stackSize++] = node.leftOrFirst;
        }
    }
    if (stats != nullptr)
    {
        stats->Add(local);
    }
    return hit;
}

// ---------------------------------------------------------------------------
// Top level
// ---------------------------------------------------------------------------

struct TopLevelAccelerationStructure::Instance
{
    const BottomLevelAccelerationStructure* blas = nullptr;
    math::Mat4 worldFromObject = math::Mat4::Identity();
    math::Mat4 objectFromWorld = math::Mat4::Identity();
    // Transpose of objectFromWorld. Its upper 3x3 is the inverse transpose of
    // the transform, which is what carries a normal correctly through a
    // non-uniform scale; the translation lands in the w row and is ignored.
    math::Mat4 normalFromObject = math::Mat4::Identity();
    u32 instanceId = 0;
    u32 mask = kRayInstanceMaskAll;
    // Position in the caller's instance list, reported as RayHit::instanceIndex.
    u32 sourceIndex = 0;
};

struct TopLevelAccelerationStructure::Node
{
    math::Vec3 boundsMin{};
    u32 leftOrFirst = 0;
    math::Vec3 boundsMax{};
    u32 count = 0;
};

TopLevelAccelerationStructure::~TopLevelAccelerationStructure()
{
    Release();
}

TopLevelAccelerationStructure::TopLevelAccelerationStructure(
    TopLevelAccelerationStructure&& other) noexcept
{
    MoveFrom(other);
}

TopLevelAccelerationStructure&
TopLevelAccelerationStructure::operator=(TopLevelAccelerationStructure&& other) noexcept
{
    if (this != &other)
    {
        Release();
        MoveFrom(other);
    }
    return *this;
}

void TopLevelAccelerationStructure::Release() noexcept
{
    if (m_allocator != nullptr && m_storage != nullptr)
    {
        m_allocator->Deallocate(m_storage);
    }
    m_allocator = nullptr;
    m_storage = nullptr;
    m_storageBytes = 0;
    m_instances = nullptr;
    m_nodes = nullptr;
    m_instanceIds = nullptr;
    m_instanceCount = 0;
    m_nodeCount = 0;
    m_leafCount = 0;
    m_maxDepth = 0;
    m_triangleCount = 0;
}

void TopLevelAccelerationStructure::MoveFrom(TopLevelAccelerationStructure& other) noexcept
{
    m_allocator = std::exchange(other.m_allocator, nullptr);
    m_storage = std::exchange(other.m_storage, nullptr);
    m_storageBytes = std::exchange(other.m_storageBytes, 0);
    m_instances = std::exchange(other.m_instances, nullptr);
    m_nodes = std::exchange(other.m_nodes, nullptr);
    m_instanceIds = std::exchange(other.m_instanceIds, nullptr);
    m_instanceCount = std::exchange(other.m_instanceCount, 0);
    m_nodeCount = std::exchange(other.m_nodeCount, 0);
    m_leafCount = std::exchange(other.m_leafCount, 0);
    m_maxDepth = std::exchange(other.m_maxDepth, 0);
    m_triangleCount = std::exchange(other.m_triangleCount, 0);
}

core::Result<TopLevelAccelerationStructure, core::Error>
TopLevelAccelerationStructure::Build(memory::Span<const RayInstanceDesc> instances,
                                     memory::Allocator& allocator) noexcept
{
    using ResultType = core::Result<TopLevelAccelerationStructure, core::Error>;
    const usize count = instances.Size();
    if (count == 0 || instances.Data() == nullptr)
    {
        return ResultType::Err(ErrorCode::InvalidArgument, "top-level structure has no instances");
    }
    if (count > kMaxTlasInstances)
    {
        return ResultType::Err(ErrorCode::InvalidArgument,
                               "top-level structure has too many instances");
    }

    const u32 instanceCount = static_cast<u32>(count);
    const usize refBytes = AlignUp(sizeof(PrimitiveRef) * instanceCount, 16);
    const usize orderBytes = AlignUp(sizeof(u32) * instanceCount, 16);
    void* scratch = allocator.Allocate(refBytes + orderBytes, 16);
    if (scratch == nullptr)
    {
        return ResultType::Err(ErrorCode::Internal,
                               "top-level structure scratch allocation failed");
    }
    auto* refs = static_cast<PrimitiveRef*>(scratch);
    auto* order = reinterpret_cast<u32*>(static_cast<unsigned char*>(scratch) + refBytes);

    u32 triangleCount = 0;
    for (u32 index = 0; index < instanceCount; ++index)
    {
        const RayInstanceDesc& instance = instances[index];
        const char* failure = nullptr;
        math::Mat4 inverse{};
        if (instance.blas == nullptr || !instance.blas->IsValid())
        {
            failure = "top-level instance has no valid bottom-level structure";
        }
        else if (instance.mask == 0)
        {
            failure = "top-level instance mask is zero";
        }
        else if (!IsFinite(instance.transform) || !instance.transform.TryInverse(inverse) ||
                 !IsFinite(inverse))
        {
            failure = "top-level instance transform is not invertible";
        }
        if (failure != nullptr)
        {
            allocator.Deallocate(scratch);
            return ResultType::Err(ErrorCode::InvalidArgument, failure);
        }
        const math::Aabb local = instance.blas->Bounds();
        PrimitiveRef& ref = refs[index];
        ref.minimum = math::Vec3{kRayMaxDistance, kRayMaxDistance, kRayMaxDistance};
        ref.maximum = math::Vec3{-kRayMaxDistance, -kRayMaxDistance, -kRayMaxDistance};
        for (u32 corner = 0; corner < 8; ++corner)
        {
            const math::Vec3 point =
                TransformBoxCorner(instance.transform, local.Min(), local.Max(), corner);
            ref.minimum = math::Min(ref.minimum, point);
            ref.maximum = math::Max(ref.maximum, point);
        }
        ref.centroid = (ref.minimum + ref.maximum) * 0.5f;
        order[index] = index;
        triangleCount += instance.blas->TriangleCount();
    }

    const u32 nodeCapacity = 2 * instanceCount - 1;
    const usize instanceBytes = AlignUp(sizeof(Instance) * instanceCount, 16);
    const usize nodeBytes = AlignUp(sizeof(Node) * nodeCapacity, 16);
    const usize idBytes = AlignUp(sizeof(u32) * instanceCount, 16);
    void* storage = allocator.Allocate(instanceBytes + nodeBytes + idBytes, 16);
    if (storage == nullptr)
    {
        allocator.Deallocate(scratch);
        return ResultType::Err(ErrorCode::Internal, "top-level structure allocation failed");
    }

    TopLevelAccelerationStructure result;
    result.m_allocator = &allocator;
    result.m_storage = storage;
    result.m_storageBytes = instanceBytes + nodeBytes + idBytes;
    result.m_instances = static_cast<Instance*>(storage);
    result.m_nodes = reinterpret_cast<Node*>(static_cast<unsigned char*>(storage) + instanceBytes);
    result.m_instanceIds =
        reinterpret_cast<u32*>(static_cast<unsigned char*>(storage) + instanceBytes + nodeBytes);
    for (u32 node = 0; node < nodeCapacity; ++node)
    {
        new (&result.m_nodes[node]) Node{};
    }

    BvhBuilder<Node> builder;
    builder.primitives = refs;
    builder.order = order;
    builder.nodes = result.m_nodes;
    builder.nodeCount = 1;
    builder.maxLeafSize = 1;
    builder.binCount = 16;
    builder.Build(0, 0, instanceCount, 0);

    // Leaves reference instances by slot, so instances are stored in leaf
    // order. Each keeps the caller's index, which is what RayHit reports, so
    // the caller's indexing is unaffected by how the build ordered them.
    for (u32 slot = 0; slot < instanceCount; ++slot)
    {
        const u32 sourceIndex = order[slot];
        const RayInstanceDesc& source = instances[sourceIndex];
        Instance* instance = new (&result.m_instances[slot]) Instance{};
        instance->blas = source.blas;
        instance->worldFromObject = source.transform;
        (void)source.transform.TryInverse(instance->objectFromWorld);
        instance->normalFromObject = instance->objectFromWorld.Transpose();
        instance->instanceId = source.instanceId;
        instance->mask = source.mask;
        instance->sourceIndex = sourceIndex;
    }
    for (u32 index = 0; index < instanceCount; ++index)
    {
        result.m_instanceIds[index] = instances[index].instanceId;
    }
    allocator.Deallocate(scratch);

    result.m_instanceCount = instanceCount;
    result.m_nodeCount = builder.nodeCount;
    result.m_leafCount = builder.leafCount;
    result.m_maxDepth = builder.maxDepth;
    result.m_triangleCount = triangleCount;
    return ResultType::Ok(std::move(result));
}

bool TopLevelAccelerationStructure::IsValid() const noexcept
{
    return m_storage != nullptr && m_instanceCount > 0;
}

u32 TopLevelAccelerationStructure::InstanceCount() const noexcept
{
    return m_instanceCount;
}

u32 TopLevelAccelerationStructure::TriangleCount() const noexcept
{
    return m_triangleCount;
}

math::Aabb TopLevelAccelerationStructure::Bounds() const noexcept
{
    if (!IsValid())
    {
        return math::Aabb::Empty();
    }
    return math::Aabb::FromMinMax(m_nodes[0].boundsMin, m_nodes[0].boundsMax);
}

AccelerationStructureStats TopLevelAccelerationStructure::Stats() const noexcept
{
    AccelerationStructureStats stats{};
    stats.triangleCount = m_triangleCount;
    stats.nodeCount = m_nodeCount;
    stats.leafCount = m_leafCount;
    stats.maxDepth = m_maxDepth;
    stats.maxLeafTriangles = 0;
    return stats;
}

usize TopLevelAccelerationStructure::DataSizeBytes() const noexcept
{
    return m_storageBytes;
}

u32 TopLevelAccelerationStructure::InstanceId(u32 index) const noexcept
{
    return index < m_instanceCount ? m_instanceIds[index] : 0;
}

bool TopLevelAccelerationStructure::Intersect(const RayDesc& ray, RayHit& out, u32 instanceMask,
                                              RayTraversalStats* stats) const noexcept
{
    if (!IsValid() || !ValidRay(ray) || (instanceMask & kRayInstanceMaskAll) == 0)
    {
        return false;
    }
    const math::Vec3 invDirection = SafeInverse(ray.direction);
    RayTraversalStats local{};
    RayHit best{};
    u32 bestSlot = kInvalidRayIndex;
    f32 closest = ray.tMax;

    u32 stack[kTraversalStackSize];
    u32 stackSize = 0;
    stack[stackSize++] = 0;
    while (stackSize > 0)
    {
        const Node& node = m_nodes[stack[--stackSize]];
        ++local.nodeTests;
        if (IntersectNode(node, ray.origin, invDirection, ray.tMin, closest) > closest)
        {
            continue;
        }
        if (node.count == 0)
        {
            if (stackSize + 2 <= kTraversalStackSize)
            {
                stack[stackSize++] = node.leftOrFirst + 1;
                stack[stackSize++] = node.leftOrFirst;
            }
            continue;
        }
        for (u32 slot = node.leftOrFirst; slot < node.leftOrFirst + node.count; ++slot)
        {
            const Instance& instance = m_instances[slot];
            if ((instance.mask & instanceMask) == 0)
            {
                continue;
            }
            ++local.instanceTests;
            // An affine transform preserves the ray parameter as long as the
            // direction is not renormalized, so tMin and the running closest
            // distance carry over unchanged.
            RayDesc objectRay{};
            objectRay.origin = instance.objectFromWorld.TransformPoint(ray.origin);
            objectRay.direction = instance.objectFromWorld.TransformVector(ray.direction);
            objectRay.tMin = ray.tMin;
            objectRay.tMax = closest;
            RayHit hit{};
            if (instance.blas->Intersect(objectRay, hit, &local) && hit.distance <= closest)
            {
                closest = hit.distance;
                best = hit;
                bestSlot = slot;
            }
        }
    }
    if (stats != nullptr)
    {
        stats->Add(local);
    }
    if (bestSlot == kInvalidRayIndex)
    {
        return false;
    }

    const Instance& instance = m_instances[bestSlot];
    math::Vec3 geometric =
        math::Normalize(instance.normalFromObject.TransformVector(best.geometricNormal));
    math::Vec3 shading =
        math::Normalize(instance.normalFromObject.TransformVector(best.shadingNormal));
    // The object-space normals already face the object-space ray, and the
    // inverse transpose preserves the sign of a dot product with a transformed
    // direction, so they still face the world ray. This keeps that true under
    // rounding at grazing angles.
    if (math::Dot(geometric, ray.direction) > 0.0f)
    {
        geometric = -geometric;
    }
    if (math::Dot(shading, geometric) < 0.0f)
    {
        shading = -shading;
    }
    out = best;
    out.position = ray.origin + ray.direction * closest;
    out.geometricNormal = geometric;
    out.shadingNormal = shading;
    out.instanceIndex = instance.sourceIndex;
    out.instanceId = instance.instanceId;
    return true;
}

bool TopLevelAccelerationStructure::IntersectAny(const RayDesc& ray, u32 instanceMask,
                                                 RayTraversalStats* stats) const noexcept
{
    if (!IsValid() || !ValidRay(ray) || (instanceMask & kRayInstanceMaskAll) == 0)
    {
        return false;
    }
    const math::Vec3 invDirection = SafeInverse(ray.direction);
    RayTraversalStats local{};
    u32 stack[kTraversalStackSize];
    u32 stackSize = 0;
    stack[stackSize++] = 0;
    bool hit = false;
    while (stackSize > 0 && !hit)
    {
        const Node& node = m_nodes[stack[--stackSize]];
        ++local.nodeTests;
        if (IntersectNode(node, ray.origin, invDirection, ray.tMin, ray.tMax) > ray.tMax)
        {
            continue;
        }
        if (node.count == 0)
        {
            if (stackSize + 2 <= kTraversalStackSize)
            {
                stack[stackSize++] = node.leftOrFirst + 1;
                stack[stackSize++] = node.leftOrFirst;
            }
            continue;
        }
        for (u32 slot = node.leftOrFirst; slot < node.leftOrFirst + node.count && !hit; ++slot)
        {
            const Instance& instance = m_instances[slot];
            if ((instance.mask & instanceMask) == 0)
            {
                continue;
            }
            ++local.instanceTests;
            RayDesc objectRay{};
            objectRay.origin = instance.objectFromWorld.TransformPoint(ray.origin);
            objectRay.direction = instance.objectFromWorld.TransformVector(ray.direction);
            objectRay.tMin = ray.tMin;
            objectRay.tMax = ray.tMax;
            hit = instance.blas->IntersectAny(objectRay, &local);
        }
    }
    if (stats != nullptr)
    {
        stats->Add(local);
    }
    return hit;
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

RayHit TraceRay(const TopLevelAccelerationStructure& scene, const RayDesc& ray, u32 instanceMask,
                RayTraversalStats* stats) noexcept
{
    RayHit hit{};
    (void)scene.Intersect(ray, hit, instanceMask, stats);
    return hit;
}

bool TraceVisibility(const TopLevelAccelerationStructure& scene, const math::Vec3& from,
                     const math::Vec3& to, u32 instanceMask, f32 biasFraction,
                     RayTraversalStats* stats) noexcept
{
    const math::Vec3 direction = to - from;
    if (!IsFinite(direction) || math::LengthSquared(direction) <= 0.0f)
    {
        return true;
    }
    const f32 bias = std::clamp(math::IsFinite(biasFraction) ? biasFraction : 0.0f, 0.0f, 0.49f);
    RayDesc ray{};
    ray.origin = from;
    ray.direction = direction;
    ray.tMin = bias;
    ray.tMax = 1.0f - bias;
    return !scene.IntersectAny(ray, instanceMask, stats);
}

math::Vec3 OffsetRayOrigin(const math::Vec3& position, const math::Vec3& normal, f32 scale) noexcept
{
    const f32 magnitude =
        std::max(std::fabs(position.x), std::max(std::fabs(position.y), std::fabs(position.z)));
    const f32 offset = (1.0e-4f + magnitude * 1.0e-5f) * (math::IsFinite(scale) ? scale : 1.0f);
    return position + normal * offset;
}

}  // namespace cyber::software
