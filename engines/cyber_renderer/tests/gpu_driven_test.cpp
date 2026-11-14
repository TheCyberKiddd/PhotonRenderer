#include <cyber/jobs/job_system.h>
#include <cyber/math/aabb.h>
#include <cyber/math/constants.h>
#include <cyber/math/frustum.h>
#include <cyber/math/mat4.h>
#include <cyber/math/vec3.h>
#include <cyber/math/vec4.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/renderer/gpu_driven.h>
#include <cyber/renderer/meshlet.h>
#include <cyber/rhi/capabilities.h>
#include <cyber/scene/scene.h>
#include <cyber/test/test.h>

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <utility>

using namespace cyber::renderer;
using cyber::core::f32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::usize;
using cyber::math::Aabb;
using cyber::math::Mat4;
using cyber::math::Vec3;
using cyber::math::Vec4;
using cyber::memory::DynamicArray;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;

namespace
{

constexpr f32 kFov = cyber::math::kPi / 3.0f;
constexpr u32 kViewportWidth = 1280;
constexpr u32 kViewportHeight = 720;
constexpr f32 kNear = 0.1f;
constexpr f32 kFar = 600.0f;

// A 64-bit LCG: deterministic across compilers, unlike <random>'s
// distributions.
struct Random
{
    u64 state = 0;

    explicit Random(u64 seed) noexcept : state(seed * 0x9E3779B97F4A7C15ULL + 1) {}

    [[nodiscard]] u32 Next() noexcept
    {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<u32>(state >> 33U);
    }

    [[nodiscard]] f32 Unit() noexcept
    {
        return static_cast<f32>(Next() & 0xFFFFFFU) / static_cast<f32>(0x1000000);
    }

    [[nodiscard]] f32 Range(f32 low, f32 high) noexcept
    {
        return low + (high - low) * Unit();
    }
};

struct Camera
{
    Vec3 eye{};
    Mat4 viewProjection{};
};

[[nodiscard]] Camera MakeCamera(Vec3 eye, Vec3 target) noexcept
{
    Camera camera{};
    camera.eye = eye;
    const Mat4 view = Mat4::LookAt(eye, target, Vec3{0.0f, 1.0f, 0.0f});
    const Mat4 projection = Mat4::PerspectiveReverseZ(
        kFov, static_cast<f32>(kViewportWidth) / static_cast<f32>(kViewportHeight), kNear, kFar);
    camera.viewProjection = projection * view;
    return camera;
}

// Draw groups with one to four LODs, thresholds a decade apart so every LOD
// is selected somewhere in the world below.
[[nodiscard]] bool AddGroups(GpuDrivenScene& scene, u32 groupCount) noexcept
{
    for (u32 group = 0; group < groupCount; ++group)
    {
        const u32 lodCount = 1 + group % kGpuDrivenMaxLods;
        GpuDrawLod lods[kGpuDrivenMaxLods]{};
        for (u32 lod = 0; lod < lodCount; ++lod)
        {
            lods[lod].indexCount = 36U >> lod << 1U;
            lods[lod].firstIndex = group * 1000 + lod * 100;
            lods[lod].vertexOffset = static_cast<cyber::core::i32>(group * 24);
        }
        GpuDrawGroupDesc desc{};
        desc.lods = Span<const GpuDrawLod>(lods, lodCount);
        desc.lodPixelThresholds[0] = 120.0f;
        desc.lodPixelThresholds[1] = 30.0f;
        desc.lodPixelThresholds[2] = 6.0f;
        if (!scene.AddDrawGroup(desc))
        {
            return false;
        }
    }
    return true;
}

// A world of axis-aligned boxes in front of, around, and behind the camera,
// recorded as a translation and an extent so Phase 22's `TransformBounds`
// reproduces the exact same world bounds: `FromCenterExtents(t, e)`.
struct World
{
    explicit World(cyber::memory::Allocator& allocator) noexcept
        : scene(allocator), translations(allocator), extents(allocator)
    {
    }

    GpuDrivenScene scene;
    DynamicArray<Vec3> translations;
    DynamicArray<Vec3> extents;
    u32 groupCount = 0;
};

[[nodiscard]] bool BuildWorld(World& world, u32 count, u32 groupCount, u64 seed) noexcept
{
    Random random(seed);
    world.groupCount = groupCount;
    if (!AddGroups(world.scene, groupCount) || !world.scene.Reserve(count) ||
        !world.translations.TryReserve(count) || !world.extents.TryReserve(count))
    {
        return false;
    }
    for (u32 index = 0; index < count; ++index)
    {
        const Vec3 translation{random.Range(-300.0f, 300.0f), random.Range(-60.0f, 60.0f),
                               random.Range(-500.0f, 100.0f)};
        const f32 size = random.Range(0.05f, 3.0f);
        const Vec3 extent{size, size * random.Range(0.3f, 1.5f), size * random.Range(0.3f, 1.5f)};
        (void)world.translations.TryPushBack(translation);
        (void)world.extents.TryPushBack(extent);
        if (!world.scene.AddInstance(index % groupCount,
                                     Aabb::FromCenterExtents(translation, extent), index))
        {
            return false;
        }
    }
    return static_cast<bool>(world.scene.Finalize());
}

[[nodiscard]] cyber::scene::MeshInstanceDesc MeshDesc(const World& world, u32 index) noexcept
{
    const GpuDrawGroup& group = world.scene.Groups()[index % world.groupCount];
    cyber::scene::MeshInstanceDesc desc{};
    desc.meshId = 1 + index % world.groupCount;
    desc.materialId = 1;
    desc.transform = Mat4::Translation(world.translations[index]);
    desc.localBounds = Aabb::FromCenterExtents(Vec3::Zero(), world.extents[index]);
    desc.lodCount = group.lodCount;
    for (u32 lod = 0; lod + 1 < group.lodCount; ++lod)
    {
        desc.lodPixelThresholds[lod] = group.lodPixelThresholds[lod];
    }
    return desc;
}

[[nodiscard]] f32 ReverseZDepth(f32 viewDistance) noexcept
{
    // PerspectiveReverseZ maps view distance d to near * (far - d) / (d * (far - near))
    // ... evaluated through the matrix so it rounds as the culler does.
    const Mat4 projection = Mat4::PerspectiveReverseZ(
        kFov, static_cast<f32>(kViewportWidth) / static_cast<f32>(kViewportHeight), kNear, kFar);
    const Vec4 clip = projection * Vec4{0.0f, 0.0f, -viewDistance, 1.0f};
    return clip.z / clip.w;
}

}  // namespace

// ---------------------------------------------------------------------------
// Scene description
// ---------------------------------------------------------------------------

CG_TEST(gpu_driven_scene_validates_descriptions)
{
    HeapAllocator heap;
    GpuDrivenScene scene(heap);

    GpuDrawGroupDesc empty{};
    CG_EXPECT_FALSE(scene.AddDrawGroup(empty));

    GpuDrawLod lods[5] = {{36, 0, 0}, {18, 36, 0}, {6, 54, 0}, {3, 60, 0}, {3, 63, 0}};
    GpuDrawGroupDesc tooMany{};
    tooMany.lods = Span<const GpuDrawLod>(lods, 5);
    CG_EXPECT_FALSE(scene.AddDrawGroup(tooMany));

    GpuDrawGroupDesc rising{};
    rising.lods = Span<const GpuDrawLod>(lods, 3);
    rising.lodPixelThresholds[0] = 10.0f;
    rising.lodPixelThresholds[1] = 20.0f;
    CG_EXPECT_FALSE(scene.AddDrawGroup(rising));

    GpuDrawLod noIndices{0, 0, 0};
    GpuDrawGroupDesc zero{};
    zero.lods = Span<const GpuDrawLod>(&noIndices, 1);
    CG_EXPECT_FALSE(scene.AddDrawGroup(zero));

    GpuDrawGroupDesc good{};
    good.lods = Span<const GpuDrawLod>(lods, 3);
    good.lodPixelThresholds[0] = 40.0f;
    good.lodPixelThresholds[1] = 10.0f;
    auto first = scene.AddDrawGroup(good);
    CG_REQUIRE(static_cast<bool>(first));
    GpuDrawGroupDesc single{};
    single.lods = Span<const GpuDrawLod>(lods, 1);
    auto second = scene.AddDrawGroup(single);
    CG_REQUIRE(static_cast<bool>(second));
    CG_EXPECT_EQ(scene.Batches().Size(), usize{4});

    CG_EXPECT_FALSE(scene.AddInstance(7, Aabb::FromCenterExtents(Vec3::Zero(), Vec3{1.0f}), 0));
    CG_EXPECT_FALSE(scene.AddInstance(0, Aabb::Empty(), 0));
    for (u32 index = 0; index < 5; ++index)
    {
        CG_REQUIRE(
            scene
                .AddInstance(index < 3 ? 0 : 1,
                             Aabb::FromCenterExtents(Vec3{static_cast<f32>(index)}, Vec3{1.0f}),
                             100 + index)
                .IsOk());
    }
    CG_EXPECT_FALSE(scene.IsFinalized());
    CG_REQUIRE(scene.Finalize().IsOk());
    CG_EXPECT_TRUE(scene.IsFinalized());

    // Each LOD of group 0 can hold all three of its instances; group 1's one
    // batch holds two.
    const Span<const GpuDrawBatch> batches = scene.Batches();
    CG_EXPECT_EQ(batches[0].visibleBase, u32{0});
    CG_EXPECT_EQ(batches[1].visibleBase, u32{3});
    CG_EXPECT_EQ(batches[2].visibleBase, u32{6});
    CG_EXPECT_EQ(batches[3].visibleBase, u32{9});
    CG_EXPECT_EQ(batches[0].capacity, u32{3});
    CG_EXPECT_EQ(batches[3].capacity, u32{2});
    CG_EXPECT_EQ(scene.VisibleCapacity(), u32{11});
    CG_EXPECT_EQ(scene.Groups()[0].instanceCount, u32{3});

    // Adding after finalizing un-finalizes, and the reference refuses it.
    CG_REQUIRE(scene.AddInstance(1, Aabb::FromCenterExtents(Vec3::Zero(), Vec3{1.0f}), 9).IsOk());
    CG_EXPECT_FALSE(scene.IsFinalized());
    GpuCullOutput output(heap);
    const Camera camera = MakeCamera(Vec3{0.0f, 0.0f, 10.0f}, Vec3::Zero());
    const GpuCullView view =
        MakeGpuCullView(camera.viewProjection, camera.eye, kFov, kViewportHeight, kNear);
    CG_EXPECT_FALSE(CullInstancesReference(scene, view, nullptr, output));
}

// ---------------------------------------------------------------------------
// The reference culler against Phase 22
// ---------------------------------------------------------------------------

// The roadmap's "results match the CPU culler": Phase 22's `Scene::Cull` and
// the GPU culler's reference agree on every instance's visibility and LOD.
// Both use the same planes, the same positive-vertex test, and the same LOD
// rule, so the agreement is exact, not approximate.
CG_TEST(gpu_driven_reference_matches_phase22_scene_cull)
{
    HeapAllocator heap;
    constexpr u32 kCount = 120000;
    constexpr u32 kGroups = 12;
    World world(heap);
    CG_REQUIRE(BuildWorld(world, kCount, kGroups, 7));

    auto jobsResult = cyber::jobs::JobSystem::Create(cyber::jobs::JobSystem::Desc{4}, heap);
    CG_REQUIRE(static_cast<bool>(jobsResult));
    cyber::jobs::JobSystem jobs = std::move(jobsResult.Value());

    constexpr u32 kPacket = 40000;
    auto writerResult = cyber::scene::SceneCommandWriter::Create({kCount, kPacket}, heap);
    auto sceneResult = cyber::scene::Scene::Create({kCount, kPacket}, heap);
    CG_REQUIRE(static_cast<bool>(writerResult) && static_cast<bool>(sceneResult));
    cyber::scene::SceneCommandWriter writer = std::move(writerResult.Value());
    cyber::scene::Scene phase22 = std::move(sceneResult.Value());
    DynamicArray<u32> instanceOfSlot(heap);
    CG_REQUIRE(instanceOfSlot.TryResize(kCount));
    for (u32 index = 0; index < kCount; ++index)
    {
        auto handle = writer.CreateMeshInstance(MeshDesc(world, index));
        CG_REQUIRE(static_cast<bool>(handle));
        CG_REQUIRE(handle.Value().index < kCount);
        instanceOfSlot[handle.Value().index] = index;
        if ((index + 1) % kPacket == 0 || index + 1 == kCount)
        {
            CG_REQUIRE(phase22.ApplyCommands(writer.Seal()).IsOk());
            CG_REQUIRE(writer.Commit().IsOk());
        }
    }

    const Camera cameras[] = {
        MakeCamera(Vec3{0.0f, 2.0f, 40.0f}, Vec3{0.0f, 0.0f, -100.0f}),
        MakeCamera(Vec3{-50.0f, 30.0f, 0.0f}, Vec3{80.0f, -10.0f, -200.0f}),
        MakeCamera(Vec3{10.0f, -5.0f, -150.0f}, Vec3{-20.0f, 0.0f, 50.0f}),
    };
    u32 totalVisible = 0;
    u32 lodHistogram[kGpuDrivenMaxLods]{};
    for (const Camera& camera : cameras)
    {
        const GpuCullView view =
            MakeGpuCullView(camera.viewProjection, camera.eye, kFov, kViewportHeight, kNear);
        cyber::scene::SceneView sceneView{};
        sceneView.frustum = view.frustum;
        sceneView.cameraPosition = view.cameraPosition;
        sceneView.projectionScalePixels = view.projectionScalePixels;
        sceneView.nearDistance = view.nearDistance;
        auto expected = phase22.Cull(sceneView, jobs, heap);
        CG_REQUIRE(static_cast<bool>(expected));

        GpuCullOutput serial(heap);
        GpuCullOutput parallel(heap);
        CG_REQUIRE(CullInstancesReference(world.scene, view, nullptr, serial).IsOk());
        CG_REQUIRE(CullInstancesReference(world.scene, view, nullptr, parallel, &jobs).IsOk());

        // Per-instance verdicts from the reference's visible list.
        DynamicArray<u32> lodOf(heap);
        CG_REQUIRE(lodOf.TryResize(kCount));
        constexpr u32 kHidden = 0xFFFFFFFFU;
        for (u32 index = 0; index < kCount; ++index)
        {
            lodOf[index] = kHidden;
        }
        const Span<const GpuDrawGroup> groups = world.scene.Groups();
        const Span<const GpuDrawBatch> batches = world.scene.Batches();
        for (u32 group = 0; group < groups.Size(); ++group)
        {
            for (u32 lod = 0; lod < groups[group].lodCount; ++lod)
            {
                const u32 batch = groups[group].firstBatch + lod;
                for (u32 slot = 0; slot < serial.batchCounts[batch]; ++slot)
                {
                    lodOf[serial.visibleInstances[batches[batch].visibleBase + slot]] = lod;
                }
            }
        }

        const auto entries = expected.Value().Entries();
        CG_EXPECT_EQ(static_cast<u32>(entries.Size()), serial.counters.visible);
        u32 mismatches = 0;
        DynamicArray<cyber::core::u8> seen(heap);
        CG_REQUIRE(seen.TryResize(kCount));
        for (u32 index = 0; index < kCount; ++index)
        {
            seen[index] = 0;
        }
        for (usize entry = 0; entry < entries.Size(); ++entry)
        {
            const u32 instance = instanceOfSlot[entries[entry].handle.index];
            seen[instance] = 1;
            mismatches += lodOf[instance] != entries[entry].lod ? 1U : 0U;
            if (lodOf[instance] != kHidden)
            {
                ++lodHistogram[lodOf[instance]];
            }
        }
        for (u32 index = 0; index < kCount; ++index)
        {
            mismatches += (seen[index] == 0 && lodOf[index] != kHidden) ? 1U : 0U;
        }
        CG_EXPECT_EQ(mismatches, u32{0});
        CG_EXPECT_EQ(serial.counters.visible + serial.counters.frustumCulled, kCount);

        // Threading does not change a byte of the output.
        CG_EXPECT_EQ(parallel.counters.visible, serial.counters.visible);
        CG_EXPECT_EQ(parallel.draws.Size(), serial.draws.Size());
        bool identical = parallel.visibleInstances.Size() == serial.visibleInstances.Size();
        for (usize index = 0; identical && index < serial.visibleInstances.Size(); ++index)
        {
            identical = parallel.visibleInstances[index] == serial.visibleInstances[index];
        }
        CG_EXPECT_TRUE(identical);
        totalVisible += serial.counters.visible;
    }
    std::printf("phase22 equivalence: %u visible over 3 views; LOD histogram %u/%u/%u/%u\n",
                totalVisible, lodHistogram[0], lodHistogram[1], lodHistogram[2], lodHistogram[3]);
    // The world is built so every LOD is exercised and culling is real.
    for (u32 lod = 0; lod < kGpuDrivenMaxLods; ++lod)
    {
        CG_EXPECT_TRUE(lodHistogram[lod] > 0);
    }
    CG_EXPECT_TRUE(totalVisible > 0 && totalVisible < 3 * kCount / 2);
}

// The draws the compaction pass emits: one per non-empty batch, in batch
// order, each drawing its batch's slice of the visible list.
CG_TEST(gpu_driven_reference_compacts_in_batch_order)
{
    HeapAllocator heap;
    World world(heap);
    CG_REQUIRE(BuildWorld(world, 20000, 9, 3));
    const Camera camera = MakeCamera(Vec3{0.0f, 0.0f, 50.0f}, Vec3{0.0f, 0.0f, -100.0f});
    const GpuCullView view =
        MakeGpuCullView(camera.viewProjection, camera.eye, kFov, kViewportHeight, kNear);
    GpuCullOutput output(heap);
    CG_REQUIRE(CullInstancesReference(world.scene, view, nullptr, output).IsOk());

    const Span<const GpuDrawBatch> batches = world.scene.Batches();
    u32 nonEmpty = 0;
    u32 drawn = 0;
    usize draw = 0;
    for (usize batch = 0; batch < batches.Size(); ++batch)
    {
        if (output.batchCounts[batch] == 0)
        {
            continue;
        }
        ++nonEmpty;
        CG_REQUIRE(draw < output.draws.Size());
        const cyber::rhi::DrawIndexedArguments& arguments = output.draws[draw++];
        CG_EXPECT_EQ(arguments.indexCount, batches[batch].indexCount);
        CG_EXPECT_EQ(arguments.firstIndex, batches[batch].firstIndex);
        CG_EXPECT_EQ(arguments.vertexOffset, batches[batch].vertexOffset);
        CG_EXPECT_EQ(arguments.firstInstance, batches[batch].visibleBase);
        CG_EXPECT_EQ(arguments.instanceCount, output.batchCounts[batch]);
        drawn += arguments.instanceCount;
    }
    CG_EXPECT_EQ(output.counters.drawCount, nonEmpty);
    CG_EXPECT_EQ(drawn, output.counters.visible);
    CG_EXPECT_EQ(output.counters.overflow, u32{0});
    // Twenty thousand instances, a handful of draws.
    CG_EXPECT_TRUE(output.counters.drawCount <= batches.Size());
    CG_EXPECT_TRUE(output.counters.visible > output.counters.drawCount * 10);
}

// ---------------------------------------------------------------------------
// Hi-Z
// ---------------------------------------------------------------------------

CG_TEST(gpu_driven_hiz_layout_uses_ceiling_sizes)
{
    HiZLevel levels[kHiZMaxLevels]{};
    u32 texels = 0;
    CG_EXPECT_EQ(ComputeHiZLayout(0, 4, levels, &texels), u32{0});
    CG_EXPECT_EQ(ComputeHiZLayout(1, 1, levels, &texels), u32{1});
    CG_EXPECT_EQ(texels, u32{1});
    const u32 count = ComputeHiZLayout(97, 61, levels, &texels);
    CG_REQUIRE(count == 8);
    const u32 widths[] = {97, 49, 25, 13, 7, 4, 2, 1};
    const u32 heights[] = {61, 31, 16, 8, 4, 2, 1, 1};
    u32 offset = 0;
    for (u32 level = 0; level < count; ++level)
    {
        CG_EXPECT_EQ(levels[level].width, widths[level]);
        CG_EXPECT_EQ(levels[level].height, heights[level]);
        CG_EXPECT_EQ(levels[level].offset, offset);
        offset += widths[level] * heights[level];
    }
    CG_EXPECT_EQ(texels, offset);
    // 65536 wide needs 17 levels, one more than the pyramid holds.
    CG_EXPECT_EQ(ComputeHiZLayout(65536, 1, levels, &texels), u32{0});
}

// Every texel holds exactly the farthest depth of the level-0 texels under it,
// for odd sizes where the last row and column of a level have one child.
CG_TEST(gpu_driven_hiz_texels_are_the_farthest_depth_below)
{
    HeapAllocator heap;
    constexpr u32 kWidth = 97;
    constexpr u32 kHeight = 61;
    constexpr u32 kPitch = 100;
    DynamicArray<f32> depth(heap);
    CG_REQUIRE(depth.TryResize(kPitch * kHeight));
    Random random(11);
    for (usize index = 0; index < depth.Size(); ++index)
    {
        depth[index] = random.Unit();
    }
    HiZPyramid pyramid(heap);
    CG_EXPECT_FALSE(pyramid.Build(Span<const f32>(depth.Data(), 10), kWidth, kHeight, kPitch));
    CG_REQUIRE(
        pyramid.Build(Span<const f32>(depth.Data(), depth.Size()), kWidth, kHeight, kPitch).IsOk());
    CG_REQUIRE(pyramid.LevelCount() == 8);
    u32 wrong = 0;
    for (u32 level = 0; level < pyramid.LevelCount(); ++level)
    {
        const HiZLevel shape = pyramid.Level(level);
        for (u32 y = 0; y < shape.height; ++y)
        {
            for (u32 x = 0; x < shape.width; ++x)
            {
                f32 expected = 2.0f;
                const u32 x0 = x << level;
                const u32 y0 = y << level;
                for (u32 sy = y0; sy < y0 + (1U << level) && sy < kHeight; ++sy)
                {
                    for (u32 sx = x0; sx < x0 + (1U << level) && sx < kWidth; ++sx)
                    {
                        const f32 value = depth[sy * kPitch + sx];
                        expected = value < expected ? value : expected;
                    }
                }
                wrong += pyramid.Load(level, x, y) == expected ? 0U : 1U;
            }
        }
    }
    CG_EXPECT_EQ(wrong, u32{0});
}

namespace
{

// A depth buffer of axis-aligned occluder quads facing the camera, rendered
// analytically: each pixel takes the nearest quad covering its center.
struct OccluderQuad
{
    f32 left = 0.0f;
    f32 right = 0.0f;
    f32 bottom = 0.0f;
    f32 top = 0.0f;
    // View distance of the quad's plane.
    f32 distance = 0.0f;
};

[[nodiscard]] bool RenderOccluders(Span<const OccluderQuad> quads, u32 width, u32 height,
                                   DynamicArray<f32>& depth) noexcept
{
    if (!depth.TryResize(static_cast<usize>(width) * height))
    {
        return false;
    }
    const f32 aspect = static_cast<f32>(kViewportWidth) / static_cast<f32>(kViewportHeight);
    const f32 tanHalf = std::tan(kFov * 0.5f);
    for (u32 y = 0; y < height; ++y)
    {
        for (u32 x = 0; x < width; ++x)
        {
            const f32 ndcX = ((static_cast<f32>(x) + 0.5f) / static_cast<f32>(width)) * 2.0f - 1.0f;
            const f32 ndcY =
                1.0f - ((static_cast<f32>(y) + 0.5f) / static_cast<f32>(height)) * 2.0f;
            f32 nearest = 0.0f;
            for (usize quad = 0; quad < quads.Size(); ++quad)
            {
                const OccluderQuad& q = quads[quad];
                const f32 viewX = ndcX * tanHalf * aspect * q.distance;
                const f32 viewY = ndcY * tanHalf * q.distance;
                if (viewX >= q.left && viewX <= q.right && viewY >= q.bottom && viewY <= q.top)
                {
                    const f32 value = ReverseZDepth(q.distance);
                    nearest = value > nearest ? value : nearest;
                }
            }
            depth[static_cast<usize>(y) * width + x] = nearest;
        }
    }
    return true;
}

}  // namespace

// What the occlusion test promises: a box it calls occluded is behind the
// depth buffer at every pixel its screen rectangle touches. Checked against
// level 0 directly, so a pyramid or level-selection mistake shows up here.
CG_TEST(gpu_driven_hiz_occlusion_is_conservative)
{
    HeapAllocator heap;
    constexpr u32 kWidth = 160;
    constexpr u32 kHeight = 90;
    const OccluderQuad quads[] = {
        {-8.0f, 8.0f, -5.0f, 5.0f, 30.0f},
        {-40.0f, -5.0f, -20.0f, 15.0f, 60.0f},
        {2.0f, 30.0f, -12.0f, 2.0f, 45.0f},
    };
    DynamicArray<f32> depth(heap);
    CG_REQUIRE(RenderOccluders(Span<const OccluderQuad>(quads, 3), kWidth, kHeight, depth));
    HiZPyramid pyramid(heap);
    CG_REQUIRE(pyramid.Build(Span<const f32>(depth.Data(), depth.Size()), kWidth, kHeight).IsOk());

    const Camera camera = MakeCamera(Vec3::Zero(), Vec3{0.0f, 0.0f, -1.0f});
    Random random(23);
    u32 occluded = 0;
    u32 visible = 0;
    u32 untestable = 0;
    u32 violations = 0;
    for (u32 trial = 0; trial < 20000; ++trial)
    {
        const Vec3 center{random.Range(-40.0f, 40.0f), random.Range(-25.0f, 25.0f),
                          -random.Range(1.0f, 120.0f)};
        const f32 size = random.Range(0.05f, 6.0f);
        const Aabb box = Aabb::FromCenterExtents(center, Vec3{size, size * 0.7f, size * 1.3f});
        const HiZTestResult result = TestHiZOcclusion(pyramid, camera.viewProjection, box);
        if (result.visibility == HiZVisibility::Untestable)
        {
            ++untestable;
            continue;
        }
        if (result.visibility == HiZVisibility::Visible)
        {
            ++visible;
            continue;
        }
        ++occluded;
        // Recompute the rectangle the way the test does and compare with
        // every level-0 texel inside it.
        f32 minX = 2.0f;
        f32 maxX = -2.0f;
        f32 minY = 2.0f;
        f32 maxY = -2.0f;
        for (u32 corner = 0; corner < 8; ++corner)
        {
            const Vec4 point{(corner & 1U) != 0 ? box.Max().x : box.Min().x,
                             (corner & 2U) != 0 ? box.Max().y : box.Min().y,
                             (corner & 4U) != 0 ? box.Max().z : box.Min().z, 1.0f};
            const Vec4 clip = camera.viewProjection * point;
            minX = std::fmin(minX, clip.x / clip.w);
            maxX = std::fmax(maxX, clip.x / clip.w);
            minY = std::fmin(minY, clip.y / clip.w);
            maxY = std::fmax(maxY, clip.y / clip.w);
        }
        const auto texel = [](f32 value, u32 extent) noexcept
        {
            const f32 floored = std::floor(value);
            return floored <= 0.0f ? 0U
                                   : (static_cast<u32>(floored) < extent ? static_cast<u32>(floored)
                                                                         : extent - 1);
        };
        const u32 x0 = texel((std::fmax(minX, -1.0f) * 0.5f + 0.5f) * kWidth, kWidth);
        const u32 x1 = texel((std::fmin(maxX, 1.0f) * 0.5f + 0.5f) * kWidth, kWidth);
        const u32 y0 = texel((0.5f - std::fmin(maxY, 1.0f) * 0.5f) * kHeight, kHeight);
        const u32 y1 = texel((0.5f - std::fmax(minY, -1.0f) * 0.5f) * kHeight, kHeight);
        for (u32 y = y0; y <= y1; ++y)
        {
            for (u32 x = x0; x <= x1; ++x)
            {
                violations += result.boxDepth < depth[y * kWidth + x] ? 0U : 1U;
            }
        }
    }
    std::printf("hi-z conservativeness: %u occluded, %u visible, %u untestable\n", occluded,
                visible, untestable);
    CG_EXPECT_EQ(violations, u32{0});
    // The scene is built so that every outcome happens often.
    CG_EXPECT_TRUE(occluded > 1000);
    CG_EXPECT_TRUE(visible > 1000);
    CG_EXPECT_TRUE(untestable > 100);
}

// The culler with occlusion on: a wall hides a block of instances, nothing in
// front of the wall or beside it is lost, and the same view without
// occlusion draws them all.
CG_TEST(gpu_driven_occlusion_culls_instances_behind_a_wall)
{
    HeapAllocator heap;
    constexpr u32 kWidth = 320;
    constexpr u32 kHeight = 180;
    const OccluderQuad wall[] = {{-20.0f, 20.0f, -12.0f, 12.0f, 40.0f}};
    DynamicArray<f32> depth(heap);
    CG_REQUIRE(RenderOccluders(Span<const OccluderQuad>(wall, 1), kWidth, kHeight, depth));
    HiZPyramid pyramid(heap);
    CG_REQUIRE(pyramid.Build(Span<const f32>(depth.Data(), depth.Size()), kWidth, kHeight).IsOk());

    GpuDrivenScene scene(heap);
    CG_REQUIRE(AddGroups(scene, 1));
    // Behind the wall: a 10 x 6 x 5 grid well inside its silhouette.
    u32 behind = 0;
    for (u32 z = 0; z < 5; ++z)
    {
        for (u32 y = 0; y < 6; ++y)
        {
            for (u32 x = 0; x < 10; ++x)
            {
                const Vec3 center{-9.0f + 2.0f * x, -5.0f + 2.0f * y, -60.0f - 10.0f * z};
                CG_REQUIRE(scene.AddInstance(0, Aabb::FromCenterExtents(center, Vec3{0.4f}), behind)
                               .IsOk());
                ++behind;
            }
        }
    }
    // In front of the wall, and beside it outside its shadow (the wall's
    // half-width of 20 at distance 40 grows to 25 at distance 50): never
    // occluded.
    constexpr u32 kFrontBase = 1000;
    u32 front = 0;
    for (u32 index = 0; index < 40; ++index)
    {
        const f32 step = static_cast<f32>(index % 10);
        const Vec3 center =
            index < 20 ? Vec3{-7.5f + 1.5f * step, 0.0f, -20.0f - static_cast<f32>(index % 20)}
                       : Vec3{(index < 30 ? -1.0f : 1.0f) * (28.0f + 2.0f * step), 0.0f, -50.0f};
        CG_REQUIRE(
            scene.AddInstance(0, Aabb::FromCenterExtents(center, Vec3{0.5f}), kFrontBase + index)
                .IsOk());
        ++front;
    }
    CG_REQUIRE(scene.Finalize().IsOk());

    const Camera camera = MakeCamera(Vec3::Zero(), Vec3{0.0f, 0.0f, -1.0f});
    GpuCullView view =
        MakeGpuCullView(camera.viewProjection, camera.eye, kFov, kViewportHeight, kNear);
    GpuCullOutput without(heap);
    CG_REQUIRE(CullInstancesReference(scene, view, nullptr, without).IsOk());

    view.flags |= kGpuCullOcclusion;
    GpuCullOutput with(heap);
    CG_EXPECT_FALSE(CullInstancesReference(scene, view, nullptr, with));
    CG_REQUIRE(CullInstancesReference(scene, view, &pyramid, with).IsOk());

    u32 frontVisible = 0;
    u32 behindVisible = 0;
    for (usize batch = 0; batch < with.batchCounts.Size(); ++batch)
    {
        const GpuDrawBatch& shape = scene.Batches()[batch];
        for (u32 slot = 0; slot < with.batchCounts[batch]; ++slot)
        {
            const u32 id = with.visibleInstances[shape.visibleBase + slot];
            if (id >= kFrontBase)
            {
                ++frontVisible;
            }
            else
            {
                ++behindVisible;
            }
        }
    }
    std::printf("occlusion: %u/%u behind the wall culled, %u/%u in front kept; "
                "without occlusion %u visible\n",
                with.counters.occlusionCulled, behind, frontVisible, front,
                without.counters.visible);
    CG_EXPECT_EQ(without.counters.visible, behind + front);
    CG_EXPECT_EQ(frontVisible, front);
    CG_EXPECT_EQ(behindVisible, u32{0});
    CG_EXPECT_EQ(with.counters.occlusionCulled, behind);
    CG_EXPECT_EQ(with.counters.visible + with.counters.occlusionCulled, behind + front);
}

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

CG_TEST(gpu_driven_geometry_path_falls_back_to_the_vertex_pipeline)
{
    using cyber::rhi::Capabilities;
    using cyber::rhi::Feature;
    Capabilities none{};
    Capabilities mesh{};
    mesh.features.Add(Feature::MeshShaders);

    GeometryPathPlan plan = SelectGeometryPath(mesh, GeometryPathPreference::VertexOnly);
    CG_EXPECT_EQ(plan.path, GeometryPath::VertexPipeline);
    CG_EXPECT_EQ(plan.reason, GeometryFallbackReason::PolicyDisabled);
    CG_EXPECT_TRUE(plan.meshShadersReported);

    plan = SelectGeometryPath(none, GeometryPathPreference::PreferMeshShaders);
    CG_EXPECT_EQ(plan.path, GeometryPath::VertexPipeline);
    CG_EXPECT_EQ(plan.reason, GeometryFallbackReason::Unsupported);
    CG_EXPECT_FALSE(plan.meshShadersReported);

    // A capable adapter is told the build has no mesh-shader pipeline rather
    // than being handed one that does not exist.
    plan = SelectGeometryPath(mesh, GeometryPathPreference::PreferMeshShaders);
    if (kMeshShaderPathCompiledIn)
    {
        CG_EXPECT_EQ(plan.path, GeometryPath::MeshShader);
        CG_EXPECT_EQ(plan.reason, GeometryFallbackReason::None);
    }
    else
    {
        CG_EXPECT_EQ(plan.path, GeometryPath::VertexPipeline);
        CG_EXPECT_EQ(plan.reason, GeometryFallbackReason::NotImplemented);
    }
    CG_EXPECT_STREQ(GeometryPathName(GeometryPath::MeshShader), "mesh_shader");
    CG_EXPECT_STREQ(GeometryFallbackReasonName(GeometryFallbackReason::NotImplemented),
                    "not_implemented");
}

CG_TEST(gpu_driven_async_compute_is_kept_only_when_measured)
{
    AsyncComputePolicyDesc policy{};
    AsyncComputePlan plan = SelectAsyncComputePlan(policy, true, nullptr);
    CG_EXPECT_FALSE(plan.useAsyncCompute);
    CG_EXPECT_EQ(plan.reason, AsyncComputeReason::NotMeasured);
    CG_EXPECT_EQ(plan.cullingQueue, cyber::rhi::QueueKind::Graphics);

    AsyncComputeMeasurement few{10.0f, 5.0f, 3};
    plan = SelectAsyncComputePlan(policy, true, &few);
    CG_EXPECT_EQ(plan.reason, AsyncComputeReason::NotMeasured);

    AsyncComputeMeasurement slower{10.0f, 10.4f, 16};
    plan = SelectAsyncComputePlan(policy, true, &slower);
    CG_EXPECT_FALSE(plan.useAsyncCompute);
    CG_EXPECT_EQ(plan.reason, AsyncComputeReason::NoMeasuredBenefit);
    CG_EXPECT_TRUE(plan.measuredGainFraction < 0.0f);

    AsyncComputeMeasurement marginal{10.0f, 9.7f, 16};
    plan = SelectAsyncComputePlan(policy, true, &marginal);
    CG_EXPECT_EQ(plan.reason, AsyncComputeReason::NoMeasuredBenefit);

    AsyncComputeMeasurement faster{10.0f, 8.0f, 16};
    plan = SelectAsyncComputePlan(policy, true, &faster);
    CG_EXPECT_TRUE(plan.useAsyncCompute);
    CG_EXPECT_EQ(plan.cullingQueue, cyber::rhi::QueueKind::Compute);
    CG_EXPECT_EQ(plan.lightCullingQueue, cyber::rhi::QueueKind::Compute);
    CG_EXPECT_NEAR(plan.measuredGainFraction, 0.2f, 1.0e-5f);

    plan = SelectAsyncComputePlan(policy, false, &faster);
    CG_EXPECT_EQ(plan.reason, AsyncComputeReason::NoComputeQueue);

    policy.preference = AsyncComputePreference::Off;
    plan = SelectAsyncComputePlan(policy, true, &faster);
    CG_EXPECT_EQ(plan.reason, AsyncComputeReason::PolicyDisabled);

    policy.preference = AsyncComputePreference::Force;
    plan = SelectAsyncComputePlan(policy, true, &slower);
    CG_EXPECT_TRUE(plan.useAsyncCompute);
    CG_EXPECT_STREQ(AsyncComputeReasonName(AsyncComputeReason::NoMeasuredBenefit),
                    "no_measured_benefit");
}

// ---------------------------------------------------------------------------
// Meshlets
// ---------------------------------------------------------------------------

namespace
{

// A UV sphere with outward-facing, counter-clockwise triangles.
[[nodiscard]] bool MakeSphere(u32 rings, u32 segments, f32 radius, Vec3 center,
                              DynamicArray<Vec3>& positions, DynamicArray<u32>& indices) noexcept
{
    positions.Clear();
    indices.Clear();
    for (u32 ring = 0; ring <= rings; ++ring)
    {
        const f32 theta = cyber::math::kPi * static_cast<f32>(ring) / static_cast<f32>(rings);
        for (u32 segment = 0; segment <= segments; ++segment)
        {
            const f32 phi =
                2.0f * cyber::math::kPi * static_cast<f32>(segment) / static_cast<f32>(segments);
            const Vec3 direction{std::sin(theta) * std::cos(phi), std::cos(theta),
                                 std::sin(theta) * std::sin(phi)};
            if (!positions.TryPushBack(center + direction * radius))
            {
                return false;
            }
        }
    }
    const u32 stride = segments + 1;
    for (u32 ring = 0; ring < rings; ++ring)
    {
        for (u32 segment = 0; segment < segments; ++segment)
        {
            const u32 a = ring * stride + segment;
            const u32 b = a + stride;
            const u32 c = b + 1;
            const u32 d = a + 1;
            // Outward winding for +Y-up rings with phi increasing toward +Z.
            if (ring != 0 &&
                (!indices.TryPushBack(a) || !indices.TryPushBack(d) || !indices.TryPushBack(b)))
            {
                return false;
            }
            if (ring + 1 != rings &&
                (!indices.TryPushBack(d) || !indices.TryPushBack(c) || !indices.TryPushBack(b)))
            {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] Vec3 TriangleNormal(const DynamicArray<Vec3>& positions, u32 a, u32 b, u32 c) noexcept
{
    return cyber::math::Cross(positions[b] - positions[a], positions[c] - positions[a]);
}

}  // namespace

CG_TEST(meshlet_build_respects_limits_and_keeps_every_triangle)
{
    HeapAllocator heap;
    DynamicArray<Vec3> positions(heap);
    DynamicArray<u32> indices(heap);
    CG_REQUIRE(MakeSphere(48, 96, 1.0f, Vec3::Zero(), positions, indices));
    MeshletMesh mesh(heap);
    MeshletLimits bad{};
    bad.maxVertices = 65;
    CG_EXPECT_FALSE(BuildMeshlets(Span<const Vec3>(positions.Data(), positions.Size()),
                                  Span<const u32>(indices.Data(), indices.Size()), bad, mesh));
    CG_REQUIRE(BuildMeshlets(Span<const Vec3>(positions.Data(), positions.Size()),
                             Span<const u32>(indices.Data(), indices.Size()), MeshletLimits{}, mesh)
                   .IsOk());
    u32 triangles = 0;
    for (usize index = 0; index < mesh.meshlets.Size(); ++index)
    {
        CG_EXPECT_TRUE(mesh.meshlets[index].vertexCount <= kMeshletMaxVertices);
        CG_EXPECT_TRUE(mesh.meshlets[index].triangleCount <= kMeshletMaxTriangles);
        triangles += mesh.meshlets[index].triangleCount;
    }
    CG_EXPECT_EQ(triangles, static_cast<u32>(indices.Size() / 3));

    // Expanding every meshlet gives back the original list, in order.
    DynamicArray<u32> all(heap);
    CG_REQUIRE(all.TryResize(mesh.meshlets.Size()));
    for (usize index = 0; index < all.Size(); ++index)
    {
        all[index] = static_cast<u32>(index);
    }
    DynamicArray<u32> expanded(heap);
    CG_REQUIRE(
        ExpandMeshletsToIndices(mesh, Span<const u32>(all.Data(), all.Size()), expanded).IsOk());
    CG_REQUIRE(expanded.Size() == indices.Size());
    bool identical = true;
    for (usize index = 0; index < indices.Size(); ++index)
    {
        identical = identical && expanded[index] == indices[index];
    }
    CG_EXPECT_TRUE(identical);
    // Greedy packing closes a meshlet only when the next triangle would not
    // fit: every meshlet but the last is full in triangles, or within one
    // triangle's three new vertices of full.
    u32 underfilled = 0;
    u32 vertexBound = 0;
    for (usize index = 0; index + 1 < mesh.meshlets.Size(); ++index)
    {
        const Meshlet& meshlet = mesh.meshlets[index];
        const bool triangleFull = meshlet.triangleCount == kMeshletMaxTriangles;
        const bool vertexFull = meshlet.vertexCount + 3 > kMeshletMaxVertices;
        underfilled += triangleFull || vertexFull ? 0U : 1U;
        vertexBound += vertexFull && !triangleFull ? 1U : 0U;
    }
    std::printf("meshlets: %zu triangles in %zu meshlets, %u closed by the vertex limit\n",
                indices.Size() / 3, mesh.meshlets.Size(), vertexBound);
    CG_EXPECT_EQ(underfilled, u32{0});
    const u32 outOfRange[] = {static_cast<u32>(mesh.meshlets.Size())};
    CG_EXPECT_FALSE(ExpandMeshletsToIndices(mesh, Span<const u32>(outOfRange, 1), expanded));
}

// A meshlet the cone test drops has every triangle facing away from the eye,
// and one the sphere test drops has every vertex outside a plane: checked by
// brute force over the triangles, from many cameras.
CG_TEST(meshlet_culling_is_conservative)
{
    HeapAllocator heap;
    DynamicArray<Vec3> positions(heap);
    DynamicArray<u32> indices(heap);
    CG_REQUIRE(MakeSphere(64, 128, 4.0f, Vec3{0.0f, 0.0f, -20.0f}, positions, indices));
    MeshletMesh mesh(heap);
    CG_REQUIRE(BuildMeshlets(Span<const Vec3>(positions.Data(), positions.Size()),
                             Span<const u32>(indices.Data(), indices.Size()), MeshletLimits{}, mesh)
                   .IsOk());

    Random random(5);
    u32 backface = 0;
    u32 frustum = 0;
    u32 visible = 0;
    u32 violations = 0;
    for (u32 trial = 0; trial < 64; ++trial)
    {
        const Vec3 eye{random.Range(-30.0f, 30.0f), random.Range(-30.0f, 30.0f),
                       random.Range(-50.0f, 10.0f)};
        const Vec3 target{random.Range(-15.0f, 15.0f), random.Range(-15.0f, 15.0f),
                          -20.0f + random.Range(-6.0f, 6.0f)};
        if (cyber::math::Length(eye - Vec3{0.0f, 0.0f, -20.0f}) < 6.0f)
        {
            continue;
        }
        const Camera camera = MakeCamera(eye, target);
        MeshletCullView view{};
        view.frustum = cyber::math::Frustum::FromViewProjection(camera.viewProjection);
        view.cameraPosition = eye;
        for (usize index = 0; index < mesh.meshlets.Size(); ++index)
        {
            const Meshlet& meshlet = mesh.meshlets[index];
            const MeshletVisibility result = CullMeshlet(mesh.bounds[index], view);
            if (result == MeshletVisibility::Visible)
            {
                ++visible;
                continue;
            }
            for (u32 triangle = 0; triangle < meshlet.triangleCount; ++triangle)
            {
                const u32 base = (meshlet.triangleOffset + triangle) * 3;
                const u32 a = mesh.vertices[meshlet.vertexOffset + mesh.triangles[base + 0]];
                const u32 b = mesh.vertices[meshlet.vertexOffset + mesh.triangles[base + 1]];
                const u32 c = mesh.vertices[meshlet.vertexOffset + mesh.triangles[base + 2]];
                if (result == MeshletVisibility::BackfaceCulled)
                {
                    const Vec3 normal = TriangleNormal(positions, a, b, c);
                    for (const u32 vertex : {a, b, c})
                    {
                        violations +=
                            cyber::math::Dot(normal, positions[vertex] - eye) >= 0.0f ? 0U : 1U;
                    }
                }
                else
                {
                    bool outside = false;
                    for (const cyber::math::Plane& plane : view.frustum.planes)
                    {
                        outside = outside || (cyber::math::DistanceTo(plane, positions[a]) < 0.0f &&
                                              cyber::math::DistanceTo(plane, positions[b]) < 0.0f &&
                                              cyber::math::DistanceTo(plane, positions[c]) < 0.0f);
                    }
                    violations += outside ? 0U : 1U;
                }
            }
            backface += result == MeshletVisibility::BackfaceCulled ? 1U : 0U;
            frustum += result == MeshletVisibility::FrustumCulled ? 1U : 0U;
        }
    }
    std::printf("meshlet culling: %u back-facing, %u outside, %u visible\n", backface, frustum,
                visible);
    CG_EXPECT_EQ(violations, u32{0});
    CG_EXPECT_TRUE(backface > 100);
    CG_EXPECT_TRUE(frustum > 100);
    CG_EXPECT_TRUE(visible > 100);
}

// The vertex-pipeline fallback draws every triangle that can face the camera
// inside the frustum, and drops a large share of the rest: on a sphere seen
// from outside, close to half.
CG_TEST(meshlet_fallback_keeps_every_front_facing_triangle)
{
    HeapAllocator heap;
    DynamicArray<Vec3> positions(heap);
    DynamicArray<u32> indices(heap);
    CG_REQUIRE(MakeSphere(64, 128, 4.0f, Vec3{0.0f, 0.0f, -30.0f}, positions, indices));
    MeshletMesh mesh(heap);
    CG_REQUIRE(BuildMeshlets(Span<const Vec3>(positions.Data(), positions.Size()),
                             Span<const u32>(indices.Data(), indices.Size()), MeshletLimits{}, mesh)
                   .IsOk());
    const Camera camera = MakeCamera(Vec3::Zero(), Vec3{0.0f, 0.0f, -30.0f});
    MeshletCullView view{};
    view.frustum = cyber::math::Frustum::FromViewProjection(camera.viewProjection);
    view.cameraPosition = camera.eye;
    DynamicArray<u32> survivors(heap);
    MeshletCullStats stats{};
    CG_REQUIRE(CullMeshletsReference(mesh, view, survivors, &stats).IsOk());
    DynamicArray<u32> drawn(heap);
    CG_REQUIRE(
        ExpandMeshletsToIndices(mesh, Span<const u32>(survivors.Data(), survivors.Size()), drawn)
            .IsOk());
    CG_EXPECT_EQ(static_cast<u32>(drawn.Size() / 3), stats.visibleTriangles);

    // Every front-facing triangle of the source must appear in the drawn list.
    DynamicArray<cyber::core::u8> kept(heap);
    CG_REQUIRE(kept.TryResize(indices.Size() / 3));
    for (usize index = 0; index < kept.Size(); ++index)
    {
        kept[index] = 0;
    }
    for (usize meshletIndex = 0; meshletIndex < survivors.Size(); ++meshletIndex)
    {
        const Meshlet& meshlet = mesh.meshlets[survivors[meshletIndex]];
        for (u32 triangle = 0; triangle < meshlet.triangleCount; ++triangle)
        {
            kept[meshlet.triangleOffset + triangle] = 1;
        }
    }
    u32 frontFacing = 0;
    u32 lost = 0;
    for (usize triangle = 0; triangle < kept.Size(); ++triangle)
    {
        const u32 a = indices[triangle * 3 + 0];
        const u32 b = indices[triangle * 3 + 1];
        const u32 c = indices[triangle * 3 + 2];
        const Vec3 normal = TriangleNormal(positions, a, b, c);
        if (cyber::math::Dot(normal, positions[a] - camera.eye) < 0.0f)
        {
            ++frontFacing;
            lost += kept[triangle] == 0 ? 1U : 0U;
        }
    }
    const f32 dropped = 1.0f - static_cast<f32>(drawn.Size() / 3) / static_cast<f32>(kept.Size());
    std::printf("meshlet fallback: %zu of %zu triangles drawn (%.1f%% dropped), "
                "%u front-facing\n",
                drawn.Size() / 3, kept.Size(), dropped * 100.0f, frontFacing);
    CG_EXPECT_EQ(lost, u32{0});
    CG_EXPECT_TRUE(dropped > 0.3f);
}
