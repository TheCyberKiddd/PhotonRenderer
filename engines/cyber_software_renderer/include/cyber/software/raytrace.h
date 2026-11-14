#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/math/aabb.h>
#include <cyber/math/mat4.h>
#include <cyber/math/vec2.h>
#include <cyber/math/vec3.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/shading.h>

namespace cyber::software
{

using cyber::core::f32;
using cyber::core::u32;
using cyber::core::usize;

// Ray tracing against triangle geometry, on the CPU.
//
// This is the reference traversal the rest of Phase 31 is built on: the
// irradiance bake, the ray-traced visibility queries, and the ground truth the
// acceptance suite compares against all go through it.
//
// The vocabulary is DirectX Raytracing's on purpose. A bottom-level structure
// owns triangles in object space; a top-level structure owns instances that
// place bottom-level structures in the world with a transform and an
// identifier; a query is either a closest-hit traversal or an any-hit
// traversal; an instance mask filters which instances a ray may see. A later
// hardware path replaces the traversal and keeps the shape of the code that
// calls it, and this path remains the oracle its results are checked against.
//
// Everything here is deterministic and allocation-free after the build: a
// query walks a prebuilt node array with a fixed stack and writes nothing but
// the caller's output. Queries are const and may run from any number of
// threads at once.

// Smallest ray parameter a query accepts by default. A secondary ray starting
// exactly on a surface would otherwise re-find it; this offset, scaled by the
// caller for large scenes, is what keeps a shadow ray from shadowing itself.
inline constexpr f32 kRayDefaultTMin = 1.0e-4f;
// Stands in for an unbounded ray. Finite, so arithmetic on it stays finite.
inline constexpr f32 kRayMaxDistance = 1.0e30f;
// Instance mask that sees every instance.
inline constexpr u32 kRayInstanceMaskAll = 0xFFu;
// Reported where no primitive or instance answered.
inline constexpr u32 kInvalidRayIndex = 0xFFFFFFFFu;

// Deepest node stack a traversal needs. The builder refuses a tree deeper than
// this, so a query never needs a heap stack or a recursion.
inline constexpr u32 kMaxBvhDepth = 64;
// Hard ceiling on triangles in one bottom-level structure, which bounds both
// the build's working set and the node array.
inline constexpr u32 kMaxBlasTriangles = 1u << 21;
// Hard ceiling on instances in one top-level structure.
inline constexpr u32 kMaxTlasInstances = 1u << 16;

// origin + t * direction for t in [tMin, tMax]. The direction need not be
// normalized; distances are reported in units of its length, so a normalized
// direction reports world distance. A zero or non-finite direction misses.
struct RayDesc
{
    math::Vec3 origin{};
    math::Vec3 direction{0.0f, 0.0f, -1.0f};
    f32 tMin = kRayDefaultTMin;
    f32 tMax = kRayMaxDistance;
};

// What a closest-hit traversal found.
//
// `geometricNormal` comes from the triangle's winding and is the normal to
// offset a secondary ray along. `shadingNormal` is the barycentric blend of the
// vertex normals and is the one to light with. Both are unit length, in the
// space the query ran in, and both are flipped to face the incoming ray when
// the hit is on a back face; `frontFace` records which side was hit, so a
// caller that cares can tell.
struct RayHit
{
    bool hit = false;
    f32 distance = 0.0f;
    math::Vec3 position{};
    math::Vec3 geometricNormal{0.0f, 1.0f, 0.0f};
    math::Vec3 shadingNormal{0.0f, 1.0f, 0.0f};
    math::Vec2 uv{};
    // Barycentric weights of the second and third triangle corners, as DXR
    // reports them. The first corner's weight is 1 - u - v.
    f32 barycentricU = 0.0f;
    f32 barycentricV = 0.0f;
    // Position of the instance in the list the top-level structure was built
    // from -- not its slot in the hierarchy, which the build reorders -- or
    // kInvalidRayIndex for a query made directly against a bottom-level
    // structure.
    u32 instanceIndex = kInvalidRayIndex;
    // The identifier the instance was built with, which is how a caller finds
    // the material. Zero for a bottom-level query.
    u32 instanceId = 0;
    u32 primitiveIndex = kInvalidRayIndex;
    bool frontFace = true;
};

// Traversal work, for tuning and for proving an acceleration structure is
// doing its job. A test that asserts the BVH visits fewer triangles than a
// brute-force sweep is asserting the structure exists.
struct RayTraversalStats
{
    u32 instanceTests = 0;
    u32 nodeTests = 0;
    u32 triangleTests = 0;

    void Add(const RayTraversalStats& other) noexcept
    {
        instanceTests += other.instanceTests;
        nodeTests += other.nodeTests;
        triangleTests += other.triangleTests;
    }
};

struct AccelerationStructureBuildDesc
{
    // Three vertices per triangle, in object space, as Mesh stores them. A
    // vertex count that is not a multiple of three is rejected rather than
    // truncated, because a dropped triangle is a hole in the light transport.
    memory::Span<const VertexInput> vertices{};
    // Triangles a leaf may hold, in [1, 32]. Smaller leaves mean more nodes and
    // a tighter hierarchy; the default is where the surface-area heuristic stops
    // paying for itself on the scenes this renderer renders.
    u32 maxLeafTriangles = 4;
    // Candidate split planes per axis, in [2, 64]. The build bins triangle
    // centroids and scores each bin boundary by the surface-area heuristic.
    u32 binCount = 16;
};

// Statistics of a built hierarchy, which is what a test asserts a build is
// sound by: leaves must cover every triangle exactly once and the depth must
// stay inside the traversal stack.
struct AccelerationStructureStats
{
    u32 triangleCount = 0;
    u32 nodeCount = 0;
    u32 leafCount = 0;
    u32 maxDepth = 0;
    u32 maxLeafTriangles = 0;
};

// A bounding-volume hierarchy over triangles in one object space.
//
// The build is a top-down binned surface-area-heuristic split: centroids are
// binned along the widest axes, every bin boundary is scored by the summed
// child surface area weighted by triangle count, and the best split is taken.
// When the node is within `maxLeafTriangles`, or no split scores better than
// the leaf it would replace and the leaf would hold at most 32 triangles, the
// node becomes a leaf. A node that binning cannot separate is split at its
// median centroid instead. A node at depth kMaxBvhDepth - 1 is a leaf whatever
// its size, which is the guarantee that lets a query use a fixed stack. The
// result depends only on the input, so two builds of the same mesh are
// identical.
class BottomLevelAccelerationStructure
{
public:
    BottomLevelAccelerationStructure() noexcept = default;
    ~BottomLevelAccelerationStructure();

    BottomLevelAccelerationStructure(const BottomLevelAccelerationStructure&) = delete;
    BottomLevelAccelerationStructure& operator=(const BottomLevelAccelerationStructure&) = delete;

    BottomLevelAccelerationStructure(BottomLevelAccelerationStructure&& other) noexcept;
    BottomLevelAccelerationStructure& operator=(BottomLevelAccelerationStructure&& other) noexcept;

    // Returns InvalidArgument for an empty geometry, a vertex count that is not
    // a multiple of three, a triangle count above kMaxBlasTriangles, an
    // out-of-range descriptor, or a non-finite vertex position, and Internal
    // when allocation fails.
    //
    // Degenerate triangles -- zero area, or two coincident corners -- are kept
    // in the structure and simply never hit, because dropping them would
    // renumber every primitive index after them.
    [[nodiscard]] static core::Result<BottomLevelAccelerationStructure, core::Error>
    Build(const AccelerationStructureBuildDesc& desc, memory::Allocator& allocator) noexcept;

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] u32 TriangleCount() const noexcept;
    [[nodiscard]] math::Aabb Bounds() const noexcept;
    [[nodiscard]] AccelerationStructureStats Stats() const noexcept;
    [[nodiscard]] usize DataSizeBytes() const noexcept;

    // Closest hit in object space. `out` is written only on a hit and left
    // untouched on a miss, so a caller may keep a running nearest hit in it.
    // `stats`, when supplied, is accumulated rather than replaced.
    [[nodiscard]] bool Intersect(const RayDesc& ray, RayHit& out,
                                 RayTraversalStats* stats = nullptr) const noexcept;

    // True as soon as any hit is found inside the ray interval. This is the
    // query a visibility or shadow ray wants: it stops at the first crossing
    // instead of ordering the whole set.
    [[nodiscard]] bool IntersectAny(const RayDesc& ray,
                                    RayTraversalStats* stats = nullptr) const noexcept;

private:
    struct Triangle;
    struct Node;

    void Release() noexcept;
    void MoveFrom(BottomLevelAccelerationStructure& other) noexcept;

    memory::Allocator* m_allocator = nullptr;
    void* m_storage = nullptr;
    usize m_storageBytes = 0;
    Triangle* m_triangles = nullptr;
    Node* m_nodes = nullptr;
    u32 m_triangleCount = 0;
    u32 m_nodeCount = 0;
    u32 m_leafCount = 0;
    u32 m_maxDepth = 0;
    u32 m_maxLeafTriangles = 0;
};

// One placement of a bottom-level structure in the world.
struct RayInstanceDesc
{
    // Borrowed. It must stay alive and unmoved for as long as the top-level
    // structure that references it is used.
    const BottomLevelAccelerationStructure* blas = nullptr;
    // Object space to world space. Must be invertible; a singular transform is
    // rejected at build time rather than producing silent misses.
    math::Mat4 transform = math::Mat4::Identity();
    // Surfaces in RayHit::instanceId. The bake uses it as a material index.
    u32 instanceId = 0;
    // A ray sees this instance when its mask and this mask share a bit. The
    // default is visible to every ray; clearing a bit is how geometry is
    // excluded from, say, visibility rays without rebuilding anything.
    u32 mask = kRayInstanceMaskAll;
};

// A bounding-volume hierarchy over instances.
//
// A query transforms the ray into each candidate instance's object space,
// traverses that instance's triangles, and transforms the hit back. Transforms
// are inverted once at build time; a non-uniform scale is handled correctly
// because normals are carried through the inverse transpose rather than the
// transform.
class TopLevelAccelerationStructure
{
public:
    TopLevelAccelerationStructure() noexcept = default;
    ~TopLevelAccelerationStructure();

    TopLevelAccelerationStructure(const TopLevelAccelerationStructure&) = delete;
    TopLevelAccelerationStructure& operator=(const TopLevelAccelerationStructure&) = delete;

    TopLevelAccelerationStructure(TopLevelAccelerationStructure&& other) noexcept;
    TopLevelAccelerationStructure& operator=(TopLevelAccelerationStructure&& other) noexcept;

    // Returns InvalidArgument for an empty instance list, more than
    // kMaxTlasInstances instances, a null or invalid bottom-level structure, a
    // non-finite or singular transform, or a zero mask, and Internal when
    // allocation fails.
    [[nodiscard]] static core::Result<TopLevelAccelerationStructure, core::Error>
    Build(memory::Span<const RayInstanceDesc> instances, memory::Allocator& allocator) noexcept;

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] u32 InstanceCount() const noexcept;
    [[nodiscard]] u32 TriangleCount() const noexcept;
    [[nodiscard]] math::Aabb Bounds() const noexcept;
    [[nodiscard]] AccelerationStructureStats Stats() const noexcept;
    [[nodiscard]] usize DataSizeBytes() const noexcept;
    // Identifier the instance at this position of the build list was built
    // with, or 0 for an out-of-range index.
    [[nodiscard]] u32 InstanceId(u32 index) const noexcept;

    // Closest hit in world space. `out` is written only on a hit.
    [[nodiscard]] bool Intersect(const RayDesc& ray, RayHit& out,
                                 u32 instanceMask = kRayInstanceMaskAll,
                                 RayTraversalStats* stats = nullptr) const noexcept;

    [[nodiscard]] bool IntersectAny(const RayDesc& ray, u32 instanceMask = kRayInstanceMaskAll,
                                    RayTraversalStats* stats = nullptr) const noexcept;

private:
    struct Instance;
    struct Node;

    void Release() noexcept;
    void MoveFrom(TopLevelAccelerationStructure& other) noexcept;

    memory::Allocator* m_allocator = nullptr;
    void* m_storage = nullptr;
    usize m_storageBytes = 0;
    Instance* m_instances = nullptr;
    Node* m_nodes = nullptr;
    // Instance identifiers in the caller's order, for InstanceId().
    u32* m_instanceIds = nullptr;
    u32 m_instanceCount = 0;
    u32 m_nodeCount = 0;
    u32 m_leafCount = 0;
    u32 m_maxDepth = 0;
    u32 m_triangleCount = 0;
};

// Closest hit, as a value. A miss returns a hit with `hit` false.
[[nodiscard]] RayHit TraceRay(const TopLevelAccelerationStructure& scene, const RayDesc& ray,
                              u32 instanceMask = kRayInstanceMaskAll,
                              RayTraversalStats* stats = nullptr) noexcept;

// True when nothing blocks the segment between two world positions.
//
// The segment is shortened at both ends by `biasFraction` of its length, so a
// query between two surface points does not report the surfaces themselves as
// occluders. A zero-length segment is unoccluded.
[[nodiscard]] bool TraceVisibility(const TopLevelAccelerationStructure& scene,
                                   const math::Vec3& from, const math::Vec3& to,
                                   u32 instanceMask = kRayInstanceMaskAll,
                                   f32 biasFraction = 1.0e-3f,
                                   RayTraversalStats* stats = nullptr) noexcept;

// Ray origin offset off a surface along its geometric normal, scaled by the
// magnitude of the position so the offset stays meaningful far from the origin.
// This is what a secondary ray should start from.
[[nodiscard]] math::Vec3 OffsetRayOrigin(const math::Vec3& position, const math::Vec3& normal,
                                         f32 scale = 1.0f) noexcept;

}  // namespace cyber::software
