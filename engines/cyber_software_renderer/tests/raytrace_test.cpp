#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/raytrace.h>
#include <cyber/test/test.h>

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

using namespace cyber::software;
using cyber::core::f32;
using cyber::core::u32;
using cyber::math::Mat4;
using cyber::math::Vec2;
using cyber::math::Vec3;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;

namespace
{

VertexInput Vertex(const Vec3& position, const Vec3& normal, const Vec2& uv = Vec2{})
{
    VertexInput vertex{};
    vertex.position = position;
    vertex.normal = normal;
    vertex.uv = uv;
    return vertex;
}

void AddTriangle(std::vector<VertexInput>& vertices, const Vec3& a, const Vec3& b, const Vec3& c)
{
    const Vec3 normal = cyber::math::Normalize(cyber::math::Cross(b - a, c - a));
    vertices.push_back(Vertex(a, normal));
    vertices.push_back(Vertex(b, normal));
    vertices.push_back(Vertex(c, normal));
}

// Counter-based generator, so the soup and the rays are the same on every
// platform and every run.
struct Lcg
{
    u32 state = 0x12345678u;

    f32 Next() noexcept
    {
        state = state * 1664525u + 1013904223u;
        return static_cast<f32>(state >> 8u) * (1.0f / 16777216.0f);
    }

    f32 Range(f32 low, f32 high) noexcept
    {
        return low + (high - low) * Next();
    }
};

std::vector<VertexInput> TriangleSoup(u32 count, u32 seed)
{
    Lcg random{seed};
    std::vector<VertexInput> vertices;
    vertices.reserve(count * 3);
    for (u32 index = 0; index < count; ++index)
    {
        const Vec3 center{random.Range(-10.0f, 10.0f), random.Range(-10.0f, 10.0f),
                          random.Range(-10.0f, 10.0f)};
        const Vec3 a = center + Vec3{random.Range(-0.6f, 0.6f), random.Range(-0.6f, 0.6f),
                                     random.Range(-0.6f, 0.6f)};
        const Vec3 b = center + Vec3{random.Range(-0.6f, 0.6f), random.Range(-0.6f, 0.6f),
                                     random.Range(-0.6f, 0.6f)};
        const Vec3 c = center + Vec3{random.Range(-0.6f, 0.6f), random.Range(-0.6f, 0.6f),
                                     random.Range(-0.6f, 0.6f)};
        AddTriangle(vertices, a, b, c);
    }
    return vertices;
}

// Brute-force closest hit in double precision: the oracle the hierarchy has
// to agree with.
bool BruteForce(const std::vector<VertexInput>& vertices, const RayDesc& ray, f32& distance,
                u32& primitive)
{
    double best = static_cast<double>(ray.tMax);
    bool found = false;
    for (size_t index = 0; index + 2 < vertices.size(); index += 3)
    {
        const Vec3 p0 = vertices[index].position;
        const Vec3 e1 = vertices[index + 1].position - p0;
        const Vec3 e2 = vertices[index + 2].position - p0;
        const Vec3 p = cyber::math::Cross(ray.direction, e2);
        const double det = static_cast<double>(cyber::math::Dot(e1, p));
        if (std::fabs(det) < 1.0e-12)
        {
            continue;
        }
        const Vec3 s = ray.origin - p0;
        const double u = static_cast<double>(cyber::math::Dot(s, p)) / det;
        if (u < 0.0 || u > 1.0)
        {
            continue;
        }
        const Vec3 q = cyber::math::Cross(s, e1);
        const double v = static_cast<double>(cyber::math::Dot(ray.direction, q)) / det;
        if (v < 0.0 || u + v > 1.0)
        {
            continue;
        }
        const double t = static_cast<double>(cyber::math::Dot(e2, q)) / det;
        if (t >= static_cast<double>(ray.tMin) && t <= best)
        {
            best = t;
            primitive = static_cast<u32>(index / 3);
            found = true;
        }
    }
    distance = static_cast<f32>(best);
    return found;
}

BottomLevelAccelerationStructure BuildBlas(const std::vector<VertexInput>& vertices,
                                           HeapAllocator& allocator, u32 leafSize = 4)
{
    AccelerationStructureBuildDesc desc{};
    desc.vertices = Span<const VertexInput>(vertices.data(), vertices.size());
    desc.maxLeafTriangles = leafSize;
    auto built = BottomLevelAccelerationStructure::Build(desc, allocator);
    if (built.IsError())
    {
        return BottomLevelAccelerationStructure{};
    }
    return std::move(built).Value();
}

RayDesc Ray(const Vec3& origin, const Vec3& direction)
{
    RayDesc ray{};
    ray.origin = origin;
    ray.direction = direction;
    return ray;
}

std::vector<VertexInput> UnitQuadXY()
{
    // z = 0, facing +Z, spanning [-1, 1]^2.
    std::vector<VertexInput> vertices;
    vertices.push_back(Vertex({-1.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}));
    vertices.push_back(Vertex({1.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}));
    vertices.push_back(Vertex({1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}));
    vertices.push_back(Vertex({-1.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}));
    vertices.push_back(Vertex({1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}));
    vertices.push_back(Vertex({-1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}));
    return vertices;
}

}  // namespace

CG_TEST(software_raytrace_single_triangle_hit_attributes)
{
    HeapAllocator allocator;
    std::vector<VertexInput> vertices;
    vertices.push_back(Vertex({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}));
    vertices.push_back(Vertex({2.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}));
    vertices.push_back(Vertex({0.0f, 2.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}));
    const BottomLevelAccelerationStructure blas = BuildBlas(vertices, allocator);
    CG_REQUIRE(blas.IsValid());
    CG_EXPECT_EQ(blas.TriangleCount(), u32{1});

    RayHit hit{};
    CG_REQUIRE(blas.Intersect(Ray({0.5f, 0.5f, 3.0f}, {0.0f, 0.0f, -1.0f}), hit));
    CG_EXPECT_TRUE(hit.hit);
    CG_EXPECT_NEAR(hit.distance, 3.0f, 1.0e-5f);
    CG_EXPECT_NEAR(hit.position.x, 0.5f, 1.0e-5f);
    CG_EXPECT_NEAR(hit.position.z, 0.0f, 1.0e-5f);
    CG_EXPECT_NEAR(hit.barycentricU, 0.25f, 1.0e-5f);
    CG_EXPECT_NEAR(hit.barycentricV, 0.25f, 1.0e-5f);
    CG_EXPECT_NEAR(hit.uv.x, 0.25f, 1.0e-5f);
    CG_EXPECT_NEAR(hit.uv.y, 0.25f, 1.0e-5f);
    CG_EXPECT_NEAR(hit.geometricNormal.z, 1.0f, 1.0e-5f);
    CG_EXPECT_NEAR(hit.shadingNormal.z, 1.0f, 1.0e-5f);
    CG_EXPECT_TRUE(hit.frontFace);
    CG_EXPECT_EQ(hit.primitiveIndex, u32{0});
    CG_EXPECT_EQ(hit.instanceIndex, kInvalidRayIndex);

    // From behind: the normals flip to face the ray, and the hit says so.
    RayHit back{};
    CG_REQUIRE(blas.Intersect(Ray({0.5f, 0.5f, -2.0f}, {0.0f, 0.0f, 1.0f}), back));
    CG_EXPECT_FALSE(back.frontFace);
    CG_EXPECT_NEAR(back.geometricNormal.z, -1.0f, 1.0e-5f);
    CG_EXPECT_NEAR(back.shadingNormal.z, -1.0f, 1.0e-5f);

    // Misses: outside the triangle, pointing away, and beyond tMax.
    RayHit untouched{};
    untouched.distance = 123.0f;
    CG_EXPECT_FALSE(blas.Intersect(Ray({1.5f, 1.5f, 3.0f}, {0.0f, 0.0f, -1.0f}), untouched));
    CG_EXPECT_FALSE(blas.Intersect(Ray({0.5f, 0.5f, 3.0f}, {0.0f, 0.0f, 1.0f}), untouched));
    RayDesc shortRay = Ray({0.5f, 0.5f, 3.0f}, {0.0f, 0.0f, -1.0f});
    shortRay.tMax = 2.5f;
    CG_EXPECT_FALSE(blas.Intersect(shortRay, untouched));
    // A miss leaves the output alone, so a caller may keep a running best.
    CG_EXPECT_EQ(untouched.distance, 123.0f);
    CG_EXPECT_FALSE(untouched.hit);

    // An unnormalized direction reports distance in units of its length.
    RayHit scaled{};
    CG_REQUIRE(blas.Intersect(Ray({0.5f, 0.5f, 3.0f}, {0.0f, 0.0f, -2.0f}), scaled));
    CG_EXPECT_NEAR(scaled.distance, 1.5f, 1.0e-5f);
    CG_EXPECT_NEAR(scaled.position.z, 0.0f, 1.0e-5f);

    // Invalid rays miss rather than produce garbage.
    CG_EXPECT_FALSE(blas.Intersect(Ray({0.5f, 0.5f, 3.0f}, {0.0f, 0.0f, 0.0f}), untouched));
    CG_EXPECT_FALSE(blas.Intersect(
        Ray({std::numeric_limits<f32>::quiet_NaN(), 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}), untouched));
    RayDesc inverted = Ray({0.5f, 0.5f, 3.0f}, {0.0f, 0.0f, -1.0f});
    inverted.tMin = 5.0f;
    inverted.tMax = 1.0f;
    CG_EXPECT_FALSE(blas.Intersect(inverted, untouched));
    CG_EXPECT_FALSE(blas.IntersectAny(inverted));
}

CG_TEST(software_raytrace_build_validation)
{
    HeapAllocator allocator;
    AccelerationStructureBuildDesc desc{};
    CG_EXPECT_TRUE(BottomLevelAccelerationStructure::Build(desc, allocator).IsError());

    std::vector<VertexInput> four = UnitQuadXY();
    four.resize(4);
    desc.vertices = Span<const VertexInput>(four.data(), four.size());
    CG_EXPECT_TRUE(BottomLevelAccelerationStructure::Build(desc, allocator).IsError());

    std::vector<VertexInput> quad = UnitQuadXY();
    desc.vertices = Span<const VertexInput>(quad.data(), quad.size());
    desc.maxLeafTriangles = 0;
    CG_EXPECT_TRUE(BottomLevelAccelerationStructure::Build(desc, allocator).IsError());
    desc.maxLeafTriangles = 33;
    CG_EXPECT_TRUE(BottomLevelAccelerationStructure::Build(desc, allocator).IsError());
    desc.maxLeafTriangles = 4;
    desc.binCount = 1;
    CG_EXPECT_TRUE(BottomLevelAccelerationStructure::Build(desc, allocator).IsError());
    desc.binCount = 65;
    CG_EXPECT_TRUE(BottomLevelAccelerationStructure::Build(desc, allocator).IsError());
    desc.binCount = 16;

    quad[2].position.x = std::numeric_limits<f32>::infinity();
    CG_EXPECT_TRUE(BottomLevelAccelerationStructure::Build(desc, allocator).IsError());
    quad[2].position.x = 1.0f;
    auto built = BottomLevelAccelerationStructure::Build(desc, allocator);
    CG_REQUIRE(built.IsOk());
    CG_EXPECT_TRUE(built.Value().IsValid());
}

// The hierarchy is an accelerator, not an approximation: every closest hit must
// match a brute-force sweep, and the sweep must do far more work.
CG_TEST(software_raytrace_bvh_matches_brute_force)
{
    HeapAllocator allocator;
    const std::vector<VertexInput> soup = TriangleSoup(3000, 7u);
    const BottomLevelAccelerationStructure blas = BuildBlas(soup, allocator);
    CG_REQUIRE(blas.IsValid());

    const AccelerationStructureStats stats = blas.Stats();
    CG_EXPECT_EQ(stats.triangleCount, u32{3000});
    // A binary tree with L leaves has 2L - 1 nodes.
    CG_EXPECT_EQ(stats.nodeCount, 2 * stats.leafCount - 1);
    CG_EXPECT_TRUE(stats.maxDepth < kMaxBvhDepth);
    CG_EXPECT_TRUE(stats.maxLeafTriangles <= 32);
    CG_EXPECT_TRUE(stats.maxDepth >= 8);
    CG_EXPECT_TRUE(blas.DataSizeBytes() > 0);

    Lcg random{99u};
    RayTraversalStats traversal{};
    u32 hits = 0;
    u32 mismatches = 0;
    u32 anyMismatches = 0;
    constexpr u32 kRays = 2000;
    for (u32 index = 0; index < kRays; ++index)
    {
        const Vec3 origin{random.Range(-14.0f, 14.0f), random.Range(-14.0f, 14.0f),
                          random.Range(-14.0f, 14.0f)};
        const Vec3 target{random.Range(-8.0f, 8.0f), random.Range(-8.0f, 8.0f),
                          random.Range(-8.0f, 8.0f)};
        const RayDesc ray = Ray(origin, cyber::math::Normalize(target - origin));
        f32 expectedDistance = 0.0f;
        u32 expectedPrimitive = kInvalidRayIndex;
        const bool expected = BruteForce(soup, ray, expectedDistance, expectedPrimitive);
        RayHit hit{};
        const bool actual = blas.Intersect(ray, hit, &traversal);
        if (expected != actual ||
            (actual && std::fabs(hit.distance - expectedDistance) > 1.0e-3f * expectedDistance))
        {
            ++mismatches;
        }
        if (actual != blas.IntersectAny(ray))
        {
            ++anyMismatches;
        }
        hits += actual ? 1u : 0u;
    }
    CG_EXPECT_EQ(mismatches, u32{0});
    CG_EXPECT_EQ(anyMismatches, u32{0});
    // The soup is dense enough that most rays hit something.
    CG_EXPECT_TRUE(hits > kRays / 2);
    // A sweep tests every triangle for every ray; the hierarchy must test
    // fewer than 5% of them on average.
    const f32 meanTriangleTests =
        static_cast<f32>(traversal.triangleTests) / static_cast<f32>(kRays);
    CG_EXPECT_TRUE(meanTriangleTests < 0.05f * 3000.0f);
    CG_EXPECT_TRUE(traversal.nodeTests > 0);
}

CG_TEST(software_raytrace_build_is_deterministic)
{
    HeapAllocator allocator;
    const std::vector<VertexInput> soup = TriangleSoup(800, 3u);
    const BottomLevelAccelerationStructure first = BuildBlas(soup, allocator);
    const BottomLevelAccelerationStructure second = BuildBlas(soup, allocator);
    CG_REQUIRE(first.IsValid() && second.IsValid());
    const AccelerationStructureStats firstStats = first.Stats();
    const AccelerationStructureStats secondStats = second.Stats();
    CG_EXPECT_EQ(firstStats.nodeCount, secondStats.nodeCount);
    CG_EXPECT_EQ(firstStats.maxDepth, secondStats.maxDepth);

    Lcg random{5u};
    for (u32 index = 0; index < 300; ++index)
    {
        const RayDesc ray = Ray({random.Range(-12.0f, 12.0f), random.Range(-12.0f, 12.0f), 15.0f},
                                cyber::math::Normalize(Vec3{random.Range(-0.3f, 0.3f),
                                                            random.Range(-0.3f, 0.3f), -1.0f}));
        RayTraversalStats a{};
        RayTraversalStats b{};
        RayHit hitA{};
        RayHit hitB{};
        const bool foundA = first.Intersect(ray, hitA, &a);
        const bool foundB = second.Intersect(ray, hitB, &b);
        CG_EXPECT_EQ(foundA, foundB);
        CG_EXPECT_EQ(hitA.primitiveIndex, hitB.primitiveIndex);
        CG_EXPECT_EQ(hitA.distance, hitB.distance);
        CG_EXPECT_EQ(a.nodeTests, b.nodeTests);
        CG_EXPECT_EQ(a.triangleTests, b.triangleTests);
    }
}

// Every leaf size the builder accepts produces the same answers.
CG_TEST(software_raytrace_leaf_size_does_not_change_results)
{
    HeapAllocator allocator;
    const std::vector<VertexInput> soup = TriangleSoup(500, 11u);
    const BottomLevelAccelerationStructure fine = BuildBlas(soup, allocator, 1);
    const BottomLevelAccelerationStructure coarse = BuildBlas(soup, allocator, 32);
    CG_REQUIRE(fine.IsValid() && coarse.IsValid());
    CG_EXPECT_TRUE(fine.Stats().leafCount > coarse.Stats().leafCount);
    Lcg random{17u};
    for (u32 index = 0; index < 400; ++index)
    {
        const RayDesc ray = Ray(
            {random.Range(-12.0f, 12.0f), random.Range(-12.0f, 12.0f), random.Range(-12.0f, 12.0f)},
            cyber::math::Normalize(Vec3{random.Range(-1.0f, 1.0f), random.Range(-1.0f, 1.0f),
                                        random.Range(-1.0f, 1.0f)}));
        RayHit a{};
        RayHit b{};
        CG_EXPECT_EQ(fine.Intersect(ray, a), coarse.Intersect(ray, b));
        CG_EXPECT_EQ(a.primitiveIndex, b.primitiveIndex);
    }
}

// Coincident geometry: thousands of triangles with the same centroid cannot be
// binned apart, which is the case that exercises the median fallback and the
// depth limit.
CG_TEST(software_raytrace_degenerate_and_coincident_geometry)
{
    HeapAllocator allocator;
    std::vector<VertexInput> vertices;
    for (u32 index = 0; index < 2000; ++index)
    {
        AddTriangle(vertices, {-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    }
    // A zero-area triangle and one with two coincident corners, in the middle
    // of the list, so a dropped triangle would renumber what follows.
    AddTriangle(vertices, {5.0f, 5.0f, 0.0f}, {6.0f, 6.0f, 0.0f}, {7.0f, 7.0f, 0.0f});
    AddTriangle(vertices, {5.0f, 5.0f, 1.0f}, {5.0f, 5.0f, 1.0f}, {6.0f, 5.0f, 1.0f});
    AddTriangle(vertices, {10.0f, 10.0f, 0.0f}, {12.0f, 10.0f, 0.0f}, {10.0f, 12.0f, 0.0f});
    const BottomLevelAccelerationStructure blas = BuildBlas(vertices, allocator);
    CG_REQUIRE(blas.IsValid());
    CG_EXPECT_EQ(blas.TriangleCount(), u32{2003});
    CG_EXPECT_TRUE(blas.Stats().maxDepth < kMaxBvhDepth);

    RayHit hit{};
    CG_REQUIRE(blas.Intersect(Ray({0.0f, 0.0f, 2.0f}, {0.0f, 0.0f, -1.0f}), hit));
    CG_EXPECT_TRUE(hit.primitiveIndex < 2000);
    // The degenerate triangles are never hit.
    CG_EXPECT_FALSE(blas.IntersectAny(Ray({6.0f, 6.0f, 2.0f}, {0.0f, 0.0f, -1.0f})));
    // The last triangle keeps its index.
    CG_REQUIRE(blas.Intersect(Ray({10.5f, 10.5f, 2.0f}, {0.0f, 0.0f, -1.0f}), hit));
    CG_EXPECT_EQ(hit.primitiveIndex, u32{2002});
}

CG_TEST(software_raytrace_move_semantics)
{
    HeapAllocator allocator;
    const std::vector<VertexInput> quad = UnitQuadXY();
    BottomLevelAccelerationStructure original = BuildBlas(quad, allocator);
    CG_REQUIRE(original.IsValid());
    BottomLevelAccelerationStructure moved = std::move(original);
    CG_EXPECT_FALSE(original.IsValid());  // NOLINT(bugprone-use-after-move)
    CG_EXPECT_TRUE(moved.IsValid());
    CG_EXPECT_TRUE(moved.IntersectAny(Ray({0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f})));
    BottomLevelAccelerationStructure assigned;
    assigned = std::move(moved);
    CG_EXPECT_TRUE(assigned.IsValid());
    CG_EXPECT_FALSE(moved.IsValid());  // NOLINT(bugprone-use-after-move)
}

// Instances: transform, caller-order indices, identifiers, masks, and normals
// carried through a non-uniform scale by the inverse transpose.
CG_TEST(software_raytrace_top_level_instances)
{
    HeapAllocator allocator;
    const std::vector<VertexInput> quad = UnitQuadXY();
    const BottomLevelAccelerationStructure blas = BuildBlas(quad, allocator);
    CG_REQUIRE(blas.IsValid());

    RayInstanceDesc instances[3]{};
    // A floor at y = 0, facing +Y, scaled to 10 x 10.
    instances[0].blas = &blas;
    instances[0].transform =
        Mat4::RotationX(-cyber::math::kPi * 0.5f) * Mat4::Scale(Vec3{10.0f, 10.0f, 1.0f});
    instances[0].instanceId = 40;
    instances[0].mask = 0x01;
    // A wall at x = 5, facing -X.
    instances[1].blas = &blas;
    instances[1].transform =
        Mat4::Translation(Vec3{5.0f, 1.0f, 0.0f}) * Mat4::RotationY(-cyber::math::kPi * 0.5f);
    instances[1].instanceId = 41;
    instances[1].mask = 0x02;
    // A sheared, non-uniformly scaled quad high above, to test normals.
    instances[2].blas = &blas;
    instances[2].transform =
        Mat4::Translation(Vec3{0.0f, 20.0f, 0.0f}) *
        Mat4::RotationAxis(cyber::math::Normalize(Vec3{1.0f, 0.0f, 1.0f}), 0.7f) *
        Mat4::Scale(Vec3{4.0f, 0.5f, 1.0f});
    instances[2].instanceId = 42;
    auto built =
        TopLevelAccelerationStructure::Build(Span<const RayInstanceDesc>(instances, 3), allocator);
    CG_REQUIRE(built.IsOk());
    const TopLevelAccelerationStructure tlas = std::move(built).Value();
    CG_EXPECT_EQ(tlas.InstanceCount(), u32{3});
    CG_EXPECT_EQ(tlas.TriangleCount(), u32{6});
    CG_EXPECT_EQ(tlas.InstanceId(1), u32{41});
    CG_EXPECT_EQ(tlas.InstanceId(7), u32{0});
    CG_EXPECT_FALSE(tlas.Bounds().IsEmpty());

    RayHit floor = TraceRay(tlas, Ray({1.0f, 3.0f, 2.0f}, {0.0f, -1.0f, 0.0f}));
    CG_REQUIRE(floor.hit);
    CG_EXPECT_EQ(floor.instanceIndex, u32{0});
    CG_EXPECT_EQ(floor.instanceId, u32{40});
    CG_EXPECT_NEAR(floor.distance, 3.0f, 1.0e-4f);
    CG_EXPECT_NEAR(floor.position.y, 0.0f, 1.0e-4f);
    CG_EXPECT_NEAR(floor.geometricNormal.y, 1.0f, 1.0e-4f);
    CG_EXPECT_TRUE(floor.frontFace);

    RayHit wall = TraceRay(tlas, Ray({0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}));
    CG_REQUIRE(wall.hit);
    CG_EXPECT_EQ(wall.instanceIndex, u32{1});
    CG_EXPECT_EQ(wall.instanceId, u32{41});
    CG_EXPECT_NEAR(wall.position.x, 5.0f, 1.0e-4f);
    CG_EXPECT_NEAR(wall.geometricNormal.x, -1.0f, 1.0e-4f);

    // Masks: a ray that sees only mask 0x02 passes through the floor.
    RayHit masked{};
    CG_EXPECT_FALSE(tlas.Intersect(Ray({1.0f, 3.0f, 2.0f}, {0.0f, -1.0f, 0.0f}), masked, 0x02));
    CG_EXPECT_FALSE(tlas.IntersectAny(Ray({1.0f, 3.0f, 2.0f}, {0.0f, -1.0f, 0.0f}), 0x02));
    CG_EXPECT_TRUE(tlas.IntersectAny(Ray({1.0f, 3.0f, 2.0f}, {0.0f, -1.0f, 0.0f}), 0x01));
    CG_EXPECT_FALSE(tlas.IntersectAny(Ray({1.0f, 3.0f, 2.0f}, {0.0f, -1.0f, 0.0f}), 0x00));

    // The skewed quad: the reported normal must be perpendicular to the
    // transformed surface, which only the inverse transpose guarantees.
    const Mat4& skew = instances[2].transform;
    const Vec3 tangentU = skew.TransformVector(Vec3{1.0f, 0.0f, 0.0f});
    const Vec3 tangentV = skew.TransformVector(Vec3{0.0f, 1.0f, 0.0f});
    const Vec3 center = skew.TransformPoint(Vec3{0.1f, 0.1f, 0.0f});
    RayHit tilted = TraceRay(tlas, Ray(center + Vec3{0.0f, 5.0f, 0.0f}, {0.0f, -1.0f, 0.0f}));
    CG_REQUIRE(tilted.hit);
    CG_EXPECT_EQ(tilted.instanceId, u32{42});
    CG_EXPECT_NEAR(cyber::math::Dot(tilted.geometricNormal, cyber::math::Normalize(tangentU)), 0.0f,
                   1.0e-4f);
    CG_EXPECT_NEAR(cyber::math::Dot(tilted.geometricNormal, cyber::math::Normalize(tangentV)), 0.0f,
                   1.0e-4f);
    CG_EXPECT_NEAR(cyber::math::Length(tilted.geometricNormal), 1.0f, 1.0e-4f);
    CG_EXPECT_TRUE(cyber::math::Dot(tilted.geometricNormal, Vec3{0.0f, -1.0f, 0.0f}) < 0.0f);
    CG_EXPECT_NEAR(tilted.position.x, center.x, 1.0e-3f);
    CG_EXPECT_NEAR(tilted.position.z, center.z, 1.0e-3f);
}

CG_TEST(software_raytrace_top_level_validation)
{
    HeapAllocator allocator;
    const std::vector<VertexInput> quad = UnitQuadXY();
    const BottomLevelAccelerationStructure blas = BuildBlas(quad, allocator);
    CG_REQUIRE(blas.IsValid());

    CG_EXPECT_TRUE(
        TopLevelAccelerationStructure::Build(Span<const RayInstanceDesc>(), allocator).IsError());

    RayInstanceDesc instance{};
    CG_EXPECT_TRUE(
        TopLevelAccelerationStructure::Build(Span<const RayInstanceDesc>(&instance, 1), allocator)
            .IsError());
    const BottomLevelAccelerationStructure empty;
    instance.blas = &empty;
    CG_EXPECT_TRUE(
        TopLevelAccelerationStructure::Build(Span<const RayInstanceDesc>(&instance, 1), allocator)
            .IsError());
    instance.blas = &blas;
    instance.transform = Mat4::Scale(Vec3{1.0f, 0.0f, 1.0f});
    CG_EXPECT_TRUE(
        TopLevelAccelerationStructure::Build(Span<const RayInstanceDesc>(&instance, 1), allocator)
            .IsError());
    instance.transform = Mat4::Identity();
    instance.mask = 0;
    CG_EXPECT_TRUE(
        TopLevelAccelerationStructure::Build(Span<const RayInstanceDesc>(&instance, 1), allocator)
            .IsError());
    instance.mask = kRayInstanceMaskAll;
    CG_EXPECT_TRUE(
        TopLevelAccelerationStructure::Build(Span<const RayInstanceDesc>(&instance, 1), allocator)
            .IsOk());
}

// Many instances: the top-level hierarchy must agree with testing every
// instance in turn.
CG_TEST(software_raytrace_top_level_matches_instance_sweep)
{
    HeapAllocator allocator;
    const std::vector<VertexInput> soup = TriangleSoup(600, 23u);
    const BottomLevelAccelerationStructure blas = BuildBlas(soup, allocator);
    CG_REQUIRE(blas.IsValid());
    std::vector<RayInstanceDesc> instances(64);
    for (u32 index = 0; index < 64; ++index)
    {
        instances[index].blas = &blas;
        instances[index].transform =
            Mat4::Translation(Vec3{static_cast<f32>(index % 8) * 30.0f - 105.0f, 0.0f,
                                   static_cast<f32>(index / 8) * 30.0f - 105.0f}) *
            Mat4::RotationY(static_cast<f32>(index) * 0.37f);
        instances[index].instanceId = 1000 + index;
    }
    auto built = TopLevelAccelerationStructure::Build(
        Span<const RayInstanceDesc>(instances.data(), instances.size()), allocator);
    CG_REQUIRE(built.IsOk());
    const TopLevelAccelerationStructure tlas = std::move(built).Value();

    Lcg random{31u};
    u32 mismatches = 0;
    u32 hits = 0;
    RayTraversalStats traversal{};
    for (u32 ray = 0; ray < 1000; ++ray)
    {
        const RayDesc query =
            Ray({random.Range(-130.0f, 130.0f), 30.0f, random.Range(-130.0f, 130.0f)},
                cyber::math::Normalize(
                    Vec3{random.Range(-0.4f, 0.4f), -1.0f, random.Range(-0.4f, 0.4f)}));
        // Oracle: every instance in turn, transformed by hand.
        f32 best = query.tMax;
        u32 bestInstance = kInvalidRayIndex;
        for (u32 index = 0; index < 64; ++index)
        {
            Mat4 inverse{};
            CG_REQUIRE(instances[index].transform.TryInverse(inverse));
            RayDesc local = query;
            local.origin = inverse.TransformPoint(query.origin);
            local.direction = inverse.TransformVector(query.direction);
            local.tMax = best;
            RayHit hit{};
            if (blas.Intersect(local, hit) && hit.distance <= best)
            {
                best = hit.distance;
                bestInstance = index;
            }
        }
        RayHit hit{};
        const bool found = tlas.Intersect(query, hit, kRayInstanceMaskAll, &traversal);
        if (found != (bestInstance != kInvalidRayIndex) ||
            (found && (hit.instanceIndex != bestInstance || hit.instanceId != 1000 + bestInstance ||
                       std::fabs(hit.distance - best) > 1.0e-3f)))
        {
            ++mismatches;
        }
        hits += found ? 1u : 0u;
    }
    CG_EXPECT_EQ(mismatches, u32{0});
    CG_EXPECT_TRUE(hits > 50);
    // The top level must cull most instances: far fewer than 64 per ray.
    CG_EXPECT_TRUE(static_cast<f32>(traversal.instanceTests) / 1000.0f < 8.0f);
}

CG_TEST(software_raytrace_visibility_queries)
{
    HeapAllocator allocator;
    const std::vector<VertexInput> quad = UnitQuadXY();
    const BottomLevelAccelerationStructure blas = BuildBlas(quad, allocator);
    CG_REQUIRE(blas.IsValid());
    RayInstanceDesc instances[2]{};
    // Floor and a blocker panel at y = 2.
    instances[0].blas = &blas;
    instances[0].transform =
        Mat4::RotationX(-cyber::math::kPi * 0.5f) * Mat4::Scale(Vec3{10.0f, 10.0f, 1.0f});
    instances[1].blas = &blas;
    instances[1].transform =
        Mat4::Translation(Vec3{0.0f, 2.0f, 0.0f}) * Mat4::RotationX(-cyber::math::kPi * 0.5f);
    auto built =
        TopLevelAccelerationStructure::Build(Span<const RayInstanceDesc>(instances, 2), allocator);
    CG_REQUIRE(built.IsOk());
    const TopLevelAccelerationStructure tlas = std::move(built).Value();

    // Two points on the floor see each other: the floor itself is not an
    // occluder of a segment that lies on it, thanks to the end bias.
    CG_EXPECT_TRUE(TraceVisibility(tlas, Vec3{-4.0f, 0.0f, 4.0f}, Vec3{4.0f, 0.0f, 4.0f}));
    // A floor point under the blocker cannot see a light above it.
    CG_EXPECT_FALSE(TraceVisibility(tlas, Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.0f, 5.0f, 0.0f}));
    // One beside the blocker can.
    CG_EXPECT_TRUE(TraceVisibility(tlas, Vec3{3.0f, 0.0f, 0.0f}, Vec3{3.0f, 5.0f, 0.0f}));
    // Degenerate segments are unoccluded.
    CG_EXPECT_TRUE(TraceVisibility(tlas, Vec3{0.0f, 1.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f}));

    // The offset origin leaves the surface on the side of the normal.
    const Vec3 offset = OffsetRayOrigin(Vec3{100.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f});
    CG_EXPECT_TRUE(offset.y > 0.0f);
    CG_EXPECT_TRUE(offset.y < 0.01f);
    CG_EXPECT_TRUE(OffsetRayOrigin(Vec3{1.0e4f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f}).y > offset.y);
}
