#include <cyber/jobs/job_system.h>
#include <cyber/math/color.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/raster.h>
#include <cyber/test/test.h>

#include <cstring>
#include <utility>

using namespace cyber::software;
using namespace cyber::memory;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::usize;
using cyber::math::Color;
using cyber::math::Vec2;

namespace
{

constexpr i32 kWidth = 640;
constexpr i32 kHeight = 360;
constexpr i32 kCellsX = 100;
constexpr i32 kCellsY = 50;
constexpr usize kTriangleCount = static_cast<usize>(kCellsX) * kCellsY * 2;
constexpr u64 kPixelCount = static_cast<u64>(kWidth) * static_cast<u64>(kHeight);

Color ColorFor(u32 index)
{
    return Color{
        static_cast<f32>((index * 37u) & 255u) / 255.0f,
        static_cast<f32>((index * 91u + 17u) & 255u) / 255.0f,
        static_cast<f32>((index * 53u + 29u) & 255u) / 255.0f,
        1.0f,
    };
}

RasterTriangle MakeTriangle(const Vec2& a, const Vec2& b, const Vec2& c, const Color& ca,
                            const Color& cb, const Color& cc)
{
    RasterTriangle triangle{};
    triangle.v0 = RasterVertex{a, ca};
    triangle.v1 = RasterVertex{b, cb};
    triangle.v2 = RasterVertex{c, cc};
    return triangle;
}

}  // namespace

CG_TEST(software_raster_coverage_10k)
{
    HeapAllocator heap;

    // A jittered grid of 100 x 50 quads split into 10,000 clockwise triangles.
    // Shared grid vertices are generated once, so adjacent triangles receive
    // bit-identical positions. Jitter is bounded well below half a cell to keep
    // every quad convex, which makes the mesh a true partition of the frame.
    const usize vertexCount = static_cast<usize>(kCellsX + 1) * (kCellsY + 1);
    DynamicArray<Vec2> positions(heap);
    CG_REQUIRE(positions.TryResize(vertexCount));

    const f32 cellWidth = static_cast<f32>(kWidth) / static_cast<f32>(kCellsX);
    const f32 cellHeight = static_cast<f32>(kHeight) / static_cast<f32>(kCellsY);
    u32 random = 0x12345678u;
    auto nextRandom = [&random]() noexcept
    {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        return random;
    };
    auto jitter = [&nextRandom]() noexcept
    { return (static_cast<f32>(nextRandom() & 0xFFFFu) / 65535.0f - 0.5f) * 1.6f; };

    for (i32 row = 0; row <= kCellsY; ++row)
    {
        for (i32 column = 0; column <= kCellsX; ++column)
        {
            f32 x = static_cast<f32>(column) * cellWidth;
            f32 y = static_cast<f32>(row) * cellHeight;
            if (column > 0 && column < kCellsX)
            {
                x += jitter();
            }
            if (row > 0 && row < kCellsY)
            {
                y += jitter();
            }
            positions[static_cast<usize>(row) * (kCellsX + 1) + column] = Vec2{x, y};
        }
    }

    DynamicArray<RasterTriangle> triangles(heap);
    CG_REQUIRE(triangles.TryResize(kTriangleCount));
    usize triangleIndex = 0;
    for (i32 row = 0; row < kCellsY; ++row)
    {
        for (i32 column = 0; column < kCellsX; ++column)
        {
            const usize i00 = static_cast<usize>(row) * (kCellsX + 1) + column;
            const usize i10 = i00 + 1;
            const usize i01 = i00 + static_cast<usize>(kCellsX + 1);
            const usize i11 = i01 + 1;
            const u32 index = static_cast<u32>(triangleIndex);
            const Color c00 = ColorFor(index);
            const Color c10 = ColorFor(index + 1);
            const Color c01 = ColorFor(index + 2);
            const Color c11 = ColorFor(index + 3);
            triangles[triangleIndex++] =
                MakeTriangle(positions[i00], positions[i10], positions[i11], c00, c10, c11);
            triangles[triangleIndex++] =
                MakeTriangle(positions[i00], positions[i11], positions[i01], c00, c11, c01);
        }
    }
    CG_REQUIRE(triangleIndex == kTriangleCount);

    const Span<const RasterTriangle> mesh(triangles.Data(), triangles.Size());

    auto render = [&](cyber::jobs::JobSystem* jobSystem, i32 tileSize, Framebuffer& target,
                      RasterStats& stats)
    {
        FramebufferDesc desc{};
        desc.width = kWidth;
        desc.height = kHeight;
        auto created = Framebuffer::Create(desc, heap);
        if (!created)
        {
            return false;
        }
        target = std::move(created.Value());
        target.Clear(Color{0.0f, 0.0f, 0.0f, 0.0f});
        RasterizerDesc rasterDesc{};
        rasterDesc.tileSize = tileSize;
        const auto status = RasterizeTriangles(target, mesh, jobSystem, heap, rasterDesc, &stats);
        return static_cast<bool>(status);
    };

    auto countUncovered = [](const Framebuffer& target)
    {
        const u8* data = target.Data();
        u64 uncovered = 0;
        for (i32 y = 0; y < target.Height(); ++y)
        {
            const u8* row = data + static_cast<usize>(y) * target.StrideBytes();
            for (i32 x = 0; x < target.Width(); ++x)
            {
                if (row[static_cast<usize>(x) * 4 + 3] != 255)
                {
                    ++uncovered;
                }
            }
        }
        return uncovered;
    };

    Framebuffer single;
    RasterStats singleStats{};
    CG_REQUIRE(render(nullptr, 64, single, singleStats));
    CG_EXPECT_EQ(singleStats.triangles, static_cast<u64>(kTriangleCount));
    CG_EXPECT_EQ(singleStats.trianglesDrawn, static_cast<u64>(kTriangleCount));
    CG_EXPECT_EQ(singleStats.trianglesSkipped, u64{0});
    CG_EXPECT_EQ(singleStats.tiles, u64{60});
    CG_EXPECT_EQ(singleStats.pixels, kPixelCount);
    CG_EXPECT_EQ(countUncovered(single), u64{0});

    HeapAllocator jobHeap;
    cyber::jobs::JobSystem::Desc jobDesc{};
    jobDesc.workerCount = 4;
    auto created = cyber::jobs::JobSystem::Create(jobDesc, jobHeap);
    CG_REQUIRE(static_cast<bool>(created));
    cyber::jobs::JobSystem workers = std::move(created.Value());

    Framebuffer parallel;
    RasterStats parallelStats{};
    CG_REQUIRE(render(&workers, 64, parallel, parallelStats));
    CG_EXPECT_EQ(parallelStats.pixels, kPixelCount);
    CG_EXPECT_EQ(countUncovered(parallel), u64{0});
    CG_EXPECT_EQ(std::memcmp(single.Data(), parallel.Data(), single.DataSizeBytes()), 0);

    // Tile size is a scheduling parameter, never an image parameter.
    Framebuffer tile16;
    RasterStats tile16Stats{};
    CG_REQUIRE(render(nullptr, 16, tile16, tile16Stats));
    CG_EXPECT_EQ(tile16Stats.tiles, u64{920});
    CG_EXPECT_EQ(tile16Stats.pixels, kPixelCount);
    CG_EXPECT_EQ(std::memcmp(single.Data(), tile16.Data(), single.DataSizeBytes()), 0);

    Framebuffer tile128;
    RasterStats tile128Stats{};
    CG_REQUIRE(render(nullptr, 128, tile128, tile128Stats));
    CG_EXPECT_EQ(tile128Stats.tiles, u64{15});
    CG_EXPECT_EQ(tile128Stats.pixels, kPixelCount);
    CG_EXPECT_EQ(std::memcmp(single.Data(), tile128.Data(), single.DataSizeBytes()), 0);

    // A single worker still goes through the job system and must match too.
    HeapAllocator oneHeap;
    jobDesc.workerCount = 1;
    auto createdOne = cyber::jobs::JobSystem::Create(jobDesc, oneHeap);
    CG_REQUIRE(static_cast<bool>(createdOne));
    cyber::jobs::JobSystem oneWorker = std::move(createdOne.Value());

    Framebuffer oneWorkerTarget;
    RasterStats oneWorkerStats{};
    CG_REQUIRE(render(&oneWorker, 128, oneWorkerTarget, oneWorkerStats));
    CG_EXPECT_EQ(oneWorkerStats.pixels, kPixelCount);
    CG_EXPECT_EQ(std::memcmp(single.Data(), oneWorkerTarget.Data(), single.DataSizeBytes()), 0);
    oneWorker.Shutdown();

    // Repeated single-threaded rendering is bit-identical.
    Framebuffer repeated;
    RasterStats repeatedStats{};
    CG_REQUIRE(render(nullptr, 64, repeated, repeatedStats));
    CG_EXPECT_EQ(repeatedStats.pixels, kPixelCount);
    CG_EXPECT_EQ(std::memcmp(single.Data(), repeated.Data(), single.DataSizeBytes()), 0);

    workers.Shutdown();
}

CG_TEST(software_raster_depth_only_matches_the_color_path_and_ignores_scheduling)
{
    HeapAllocator heap;
    constexpr i32 kMapSize = 128;

    auto colorTarget =
        Framebuffer::Create(FramebufferDesc{kMapSize, kMapSize, PixelFormat::RGBA32F, 0}, heap);
    auto referenceDepth = DepthBuffer::Create(DepthBufferDesc{kMapSize, kMapSize, 0}, heap);
    auto onlyDepth = DepthBuffer::Create(DepthBufferDesc{kMapSize, kMapSize, 0}, heap);
    auto jobbedDepth = DepthBuffer::Create(DepthBufferDesc{kMapSize, kMapSize, 0}, heap);
    CG_REQUIRE(static_cast<bool>(colorTarget));
    CG_REQUIRE(static_cast<bool>(referenceDepth));
    CG_REQUIRE(static_cast<bool>(onlyDepth));
    CG_REQUIRE(static_cast<bool>(jobbedDepth));
    Framebuffer color = std::move(colorTarget.Value());
    DepthBuffer reference = std::move(referenceDepth.Value());
    DepthBuffer depthOnly = std::move(onlyDepth.Value());
    DepthBuffer jobbed = std::move(jobbedDepth.Value());

    // Overlapping triangles at different depths and perspective weights, so
    // the depth test decides the result rather than draw order alone.
    constexpr usize kCount = 24;
    DepthRasterTriangle colorTriangles[kCount]{};
    DepthOnlyRasterTriangle depthTriangles[kCount]{};
    for (usize index = 0; index < kCount; ++index)
    {
        const f32 offset = static_cast<f32>(index) * 3.7f;
        const f32 depth = 0.15f + static_cast<f32>(index % 7) * 0.11f;
        const f32 inverseW = 0.4f + static_cast<f32>(index % 5) * 0.21f;
        const Vec2 a{6.0f + offset, 4.0f + offset * 0.5f};
        const Vec2 b{96.0f - offset * 0.5f, 18.0f + offset};
        const Vec2 c{30.0f + offset, 112.0f - offset * 0.7f};
        const Vec2 points[3] = {a, b, c};
        for (u32 corner = 0; corner < 3; ++corner)
        {
            const f32 cornerDepth = depth + static_cast<f32>(corner) * 0.03f;
            const f32 cornerInverseW = inverseW + static_cast<f32>(corner) * 0.05f;
            DepthRasterVertex& colorVertex =
                corner == 0 ? colorTriangles[index].v0
                            : (corner == 1 ? colorTriangles[index].v1 : colorTriangles[index].v2);
            DepthOnlyRasterVertex& depthVertex =
                corner == 0 ? depthTriangles[index].v0
                            : (corner == 1 ? depthTriangles[index].v1 : depthTriangles[index].v2);
            colorVertex.position = points[corner];
            colorVertex.color = ColorFor(static_cast<u32>(index * 3 + corner));
            colorVertex.depth = cornerDepth;
            colorVertex.inverseW = cornerInverseW;
            depthVertex.position = points[corner];
            depthVertex.depth = cornerDepth;
            depthVertex.inverseW = cornerInverseW;
        }
    }

    color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    reference.Clear();
    RasterStats colorStats{};
    CG_REQUIRE(static_cast<bool>(RasterizeDepthTestedTriangles(
        color, reference, Span<const DepthRasterTriangle>(colorTriangles, kCount), nullptr, heap,
        {}, &colorStats)));

    depthOnly.Clear();
    RasterStats depthStats{};
    CG_REQUIRE(static_cast<bool>(RasterizeDepthOnlyTriangles(
        depthOnly, Span<const DepthOnlyRasterTriangle>(depthTriangles, kCount), nullptr, heap, {},
        &depthStats)));

    // The depth-only path must take the same coverage and depth decisions as
    // the color path, so a shadow map and a depth prepass agree exactly.
    CG_EXPECT_EQ(depthStats.triangles, colorStats.triangles);
    CG_EXPECT_EQ(depthStats.trianglesDrawn, colorStats.trianglesDrawn);
    CG_EXPECT_EQ(depthStats.pixels, colorStats.pixels);
    CG_EXPECT_EQ(depthStats.pixelsRejected, colorStats.pixelsRejected);
    CG_EXPECT_TRUE(depthStats.pixels > u64{0});

    cyber::jobs::JobSystem::Desc jobDesc{};
    jobDesc.workerCount = 4;
    auto createdJobs = cyber::jobs::JobSystem::Create(jobDesc, heap);
    CG_REQUIRE(static_cast<bool>(createdJobs));
    cyber::jobs::JobSystem workers = std::move(createdJobs.Value());

    jobbed.Clear();
    RasterStats jobbedStats{};
    RasterizerDesc jobbedDesc{};
    jobbedDesc.tileSize = 16;
    jobbedDesc.useJobs = true;
    CG_REQUIRE(static_cast<bool>(RasterizeDepthOnlyTriangles(
        jobbed, Span<const DepthOnlyRasterTriangle>(depthTriangles, kCount), &workers, heap,
        jobbedDesc, &jobbedStats)));
    CG_EXPECT_EQ(jobbedStats.pixels, depthStats.pixels);
    CG_EXPECT_EQ(jobbedStats.pixelsRejected, depthStats.pixelsRejected);

    u64 mismatches = 0;
    u64 jobbedMismatches = 0;
    for (i32 y = 0; y < kMapSize; ++y)
    {
        for (i32 x = 0; x < kMapSize; ++x)
        {
            if (depthOnly.Get(x, y) != reference.Get(x, y))
            {
                ++mismatches;
            }
            if (jobbed.Get(x, y) != depthOnly.Get(x, y))
            {
                ++jobbedMismatches;
            }
        }
    }
    CG_EXPECT_EQ(mismatches, u64{0});
    CG_EXPECT_EQ(jobbedMismatches, u64{0});
}

CG_TEST(software_raster_depth_only_rejects_invalid_input)
{
    HeapAllocator heap;
    DepthBuffer invalid{};
    const DepthOnlyRasterTriangle triangle{};
    RasterStats stats{};
    stats.pixels = 7;
    CG_EXPECT_FALSE(static_cast<bool>(RasterizeDepthOnlyTriangles(
        invalid, Span<const DepthOnlyRasterTriangle>(&triangle, 1), nullptr, heap, {}, &stats)));
    CG_EXPECT_EQ(stats.pixels, u64{0});

    auto created = DepthBuffer::Create(DepthBufferDesc{8, 8, 0}, heap);
    CG_REQUIRE(static_cast<bool>(created));
    DepthBuffer depth = std::move(created.Value());
    CG_EXPECT_FALSE(static_cast<bool>(RasterizeDepthOnlyTriangles(
        depth, Span<const DepthOnlyRasterTriangle>(nullptr, 1), nullptr, heap, {}, &stats)));

    // An empty span is a valid no-op draw.
    CG_EXPECT_TRUE(static_cast<bool>(RasterizeDepthOnlyTriangles(
        depth, Span<const DepthOnlyRasterTriangle>{}, nullptr, heap, {}, &stats)));
}
