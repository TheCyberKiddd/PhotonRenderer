#include "gi_test_scenes.h"

#include <cyber/jobs/job_system.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/software/denoise.h>
#include <cyber/software/gi_quality.h>
#include <cyber/test/test.h>

#include <cmath>
#include <cstring>
#include <utility>

using namespace cyber::software;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::math::Color;
using cyber::math::Vec3;
using cyber::memory::HeapAllocator;
using cyber::software::test::EncodeNormal;
using cyber::software::test::HashUniform;
using cyber::software::test::MakeDepth;
using cyber::software::test::MakeImage;
using cyber::software::test::ReverseZFromDistance;

namespace
{

constexpr i32 kSize = 64;
constexpr f32 kNear = 0.1f;
constexpr f32 kFar = 100.0f;

// A screen split down the middle: the left half is a near surface, the right
// half a far one, both facing the camera. The signal is 0.2 on the left and
// 0.8 on the right, plus per-pixel, per-frame uniform noise of +-amplitude.
struct SplitScene
{
    Framebuffer noisy;
    Framebuffer clean;
    Framebuffer normals;
    DepthBuffer depth;
};

bool FillSplitScene(SplitScene& scene, HeapAllocator& allocator, u32 frame, f32 amplitude,
                    f32 leftDistance = 2.0f, f32 rightDistance = 10.0f, i32 split = kSize / 2)
{
    if (!scene.noisy.IsValid())
    {
        scene.noisy = MakeImage(allocator, kSize, kSize);
        scene.clean = MakeImage(allocator, kSize, kSize);
        scene.normals = MakeImage(allocator, kSize, kSize);
        scene.depth = MakeDepth(allocator, kSize, kSize);
    }
    if (!scene.noisy.IsValid() || !scene.clean.IsValid() || !scene.normals.IsValid() ||
        !scene.depth.IsValid())
    {
        return false;
    }
    for (i32 y = 0; y < kSize; ++y)
    {
        f32* depthRow = scene.depth.Row(y);
        for (i32 x = 0; x < kSize; ++x)
        {
            const bool left = x < split;
            const f32 value = left ? 0.2f : 0.8f;
            const f32 noise =
                (HashUniform(static_cast<u32>(x), static_cast<u32>(y), frame) * 2.0f - 1.0f) *
                amplitude;
            scene.noisy.SetPixel(x, y, Color{value + noise, value + noise, value + noise, 1.0f});
            scene.clean.SetPixel(x, y, Color{value, value, value, 1.0f});
            scene.normals.SetPixel(x, y, EncodeNormal(Vec3{0.0f, 0.0f, -1.0f}));
            depthRow[x] = ReverseZFromDistance(left ? leftDistance : rightDistance, kNear, kFar);
        }
    }
    return true;
}

DenoiseGuides Guides(const SplitScene& scene, const Framebuffer* velocity = nullptr)
{
    DenoiseGuides guides{};
    guides.depth = &scene.depth;
    guides.normals = &scene.normals;
    guides.velocity = velocity;
    guides.nearPlane = kNear;
    guides.farPlane = kFar;
    return guides;
}

SpatioTemporalDenoiser MakeDenoiser(HeapAllocator& allocator)
{
    auto created = SpatioTemporalDenoiser::Create(kSize, kSize, allocator);
    if (created.IsError())
    {
        return SpatioTemporalDenoiser{};
    }
    return std::move(created).Value();
}

// Mean absolute error from the clean image over the columns either side of
// the split, which is where a filter that ignores geometry bleeds.
f32 EdgeBandError(const Framebuffer& image, const Framebuffer& clean, i32 split, i32 halfWidth)
{
    f32 sum = 0.0f;
    u32 count = 0;
    for (i32 y = 0; y < kSize; ++y)
    {
        for (i32 x = split - halfWidth; x < split + halfWidth; ++x)
        {
            sum += std::fabs(image.GetPixel(x, y).r - clean.GetPixel(x, y).r);
            ++count;
        }
    }
    return sum / static_cast<f32>(count);
}

}  // namespace

CG_TEST(software_denoise_validation)
{
    HeapAllocator allocator;
    CG_EXPECT_TRUE(SpatioTemporalDenoiser::Create(0, 4, allocator).IsError());
    CG_EXPECT_TRUE(SpatioTemporalDenoiser::Create(4, 16385, allocator).IsError());

    SplitScene scene;
    CG_REQUIRE(FillSplitScene(scene, allocator, 0, 0.1f));
    SpatioTemporalDenoiser denoiser = MakeDenoiser(allocator);
    CG_REQUIRE(denoiser.IsValid());
    CG_EXPECT_EQ(denoiser.Width(), kSize);
    CG_EXPECT_TRUE(denoiser.DataSizeBytes() > 0);
    Framebuffer output = MakeImage(allocator, kSize, kSize);
    Framebuffer small = MakeImage(allocator, 8, 8);
    const DenoiserDesc desc{};

    CG_EXPECT_TRUE(denoiser.Denoise(scene.noisy, Guides(scene), desc, scene.noisy).IsError());
    CG_EXPECT_TRUE(denoiser.Denoise(scene.noisy, Guides(scene), desc, small).IsError());
    DenoiseGuides missing = Guides(scene);
    missing.normals = nullptr;
    CG_EXPECT_TRUE(denoiser.Denoise(scene.noisy, missing, desc, output).IsError());
    DenoiseGuides badVelocity = Guides(scene, &small);
    CG_EXPECT_TRUE(denoiser.Denoise(scene.noisy, badVelocity, desc, output).IsError());
    DenoiseGuides badRange = Guides(scene);
    badRange.farPlane = badRange.nearPlane;
    CG_EXPECT_TRUE(denoiser.Denoise(scene.noisy, badRange, desc, output).IsError());
    DenoiserDesc bad = desc;
    bad.iterations = kMaxDenoiserIterations + 1;
    CG_EXPECT_TRUE(denoiser.Denoise(scene.noisy, Guides(scene), bad, output).IsError());
    bad = desc;
    bad.temporalAlpha = 0.0f;
    CG_EXPECT_TRUE(denoiser.Denoise(scene.noisy, Guides(scene), bad, output).IsError());
    bad = desc;
    bad.maxHistoryLength = 0;
    CG_EXPECT_TRUE(denoiser.Denoise(scene.noisy, Guides(scene), bad, output).IsError());

    SpatioTemporalDenoiser uninitialized;
    CG_EXPECT_TRUE(uninitialized.Denoise(scene.noisy, Guides(scene), desc, output).IsError());
    CG_EXPECT_TRUE(denoiser.Denoise(scene.noisy, Guides(scene), desc, output).IsOk());
}

// The filter is normalized: a noiseless signal passes through unchanged,
// including across the depth edge.
CG_TEST(software_denoise_preserves_clean_signal)
{
    HeapAllocator allocator;
    SplitScene scene;
    CG_REQUIRE(FillSplitScene(scene, allocator, 0, 0.0f));
    SpatioTemporalDenoiser denoiser = MakeDenoiser(allocator);
    Framebuffer output = MakeImage(allocator, kSize, kSize);
    for (u32 frame = 0; frame < 3; ++frame)
    {
        CG_REQUIRE(denoiser.Denoise(scene.noisy, Guides(scene), DenoiserDesc{}, output).IsOk());
    }
    for (i32 y = 0; y < kSize; y += 7)
    {
        for (i32 x = 0; x < kSize; ++x)
        {
            CG_EXPECT_NEAR(output.GetPixel(x, y).r, scene.clean.GetPixel(x, y).r, 1.0e-5f);
        }
    }
}

// One frame, spatial filtering only: the noise drops by a large factor, the
// mean does not move, and the depth edge stays sharp.
CG_TEST(software_denoise_spatial_reduces_noise_and_keeps_edges)
{
    HeapAllocator allocator;
    SplitScene scene;
    CG_REQUIRE(FillSplitScene(scene, allocator, 1, 0.3f));
    SpatioTemporalDenoiser denoiser = MakeDenoiser(allocator);
    Framebuffer output = MakeImage(allocator, kSize, kSize);
    DenoiserStats stats{};
    CG_REQUIRE(denoiser.Denoise(scene.noisy, Guides(scene), DenoiserDesc{}, output, &stats).IsOk());
    CG_EXPECT_EQ(stats.iterations, u32{4});
    CG_EXPECT_EQ(stats.pixelCount, static_cast<u32>(kSize * kSize));
    CG_EXPECT_EQ(stats.historyRejectedPixels, u32{0});
    CG_EXPECT_EQ(stats.spatialVariancePixels, static_cast<u32>(kSize * kSize));

    NoiseMetrics before{};
    NoiseMetrics after{};
    CG_REQUIRE(MeasureNoise(scene.noisy, scene.clean, before).IsOk());
    CG_REQUIRE(MeasureNoise(output, scene.clean, after).IsOk());
    CG_EXPECT_TRUE(after.relativeRmse < before.relativeRmse * 0.35f);
    CG_EXPECT_TRUE(std::fabs(after.relativeBias) < 0.02f);
    CG_EXPECT_TRUE(after.laplacianEnergy < before.laplacianEnergy * 0.2f);

    // The edge is preserved: the band around it is no worse than the rest of
    // the image. The control turns every edge-stopping weight off, and the two
    // sides bleed into each other.
    const f32 guided = EdgeBandError(output, scene.clean, kSize / 2, 2);
    DenoiserDesc blind{};
    blind.depthSigma = 1.0e6f;
    blind.normalPower = 0.0f;
    blind.luminanceSigma = 1.0e6f;
    SpatioTemporalDenoiser control = MakeDenoiser(allocator);
    Framebuffer blurred = MakeImage(allocator, kSize, kSize);
    CG_REQUIRE(control.Denoise(scene.noisy, Guides(scene), blind, blurred).IsOk());
    const f32 unguided = EdgeBandError(blurred, scene.clean, kSize / 2, 2);
    CG_EXPECT_TRUE(guided < 0.06f);
    CG_EXPECT_TRUE(unguided > 0.12f);
    CG_EXPECT_TRUE(unguided > 3.0f * guided);
}

// A still camera accumulates: history grows frame by frame and the result
// converges below what one frame of spatial filtering reaches.
CG_TEST(software_denoise_temporal_accumulation_converges)
{
    HeapAllocator allocator;
    SplitScene scene;
    SpatioTemporalDenoiser denoiser = MakeDenoiser(allocator);
    Framebuffer output = MakeImage(allocator, kSize, kSize);
    DenoiserDesc desc{};
    desc.spatial = false;
    NoiseMetrics first{};
    NoiseMetrics last{};
    DenoiserStats stats{};
    constexpr u32 kFrames = 16;
    for (u32 frame = 0; frame < kFrames; ++frame)
    {
        CG_REQUIRE(FillSplitScene(scene, allocator, frame + 100, 0.3f));
        CG_REQUIRE(denoiser.Denoise(scene.noisy, Guides(scene), desc, output, &stats).IsOk());
        if (frame == 0)
        {
            CG_REQUIRE(MeasureNoise(output, scene.clean, first).IsOk());
        }
    }
    CG_REQUIRE(MeasureNoise(output, scene.clean, last).IsOk());
    CG_EXPECT_NEAR(stats.meanHistoryLength, static_cast<f32>(kFrames), 1.0e-3f);
    CG_EXPECT_EQ(stats.historyRejectedPixels, u32{0});
    CG_EXPECT_EQ(stats.spatialVariancePixels, u32{0});
    // Sixteen frames averaged cut the standard deviation by about four.
    CG_EXPECT_TRUE(last.relativeRmse < first.relativeRmse * 0.35f);
    CG_EXPECT_TRUE(std::fabs(last.relativeBias) < 0.02f);

    // History length is capped and then becomes an exponential average.
    desc.maxHistoryLength = 4;
    SpatioTemporalDenoiser capped = MakeDenoiser(allocator);
    for (u32 frame = 0; frame < 8; ++frame)
    {
        CG_REQUIRE(FillSplitScene(scene, allocator, frame, 0.3f));
        CG_REQUIRE(capped.Denoise(scene.noisy, Guides(scene), desc, output, &stats).IsOk());
    }
    CG_EXPECT_NEAR(stats.meanHistoryLength, 4.0f, 1.0e-3f);

    // Reset discards history: the next frame starts over.
    denoiser.Reset();
    CG_REQUIRE(denoiser.Denoise(scene.noisy, Guides(scene), desc, output, &stats).IsOk());
    CG_EXPECT_NEAR(stats.meanHistoryLength, 1.0f, 1.0e-6f);
    CG_EXPECT_EQ(stats.historyRejectedPixels, u32{0});
}

// A disocclusion: the near surface retreats, uncovering far surface whose
// history belongs to something else. Exactly the uncovered pixels must drop
// their history, and nothing else.
CG_TEST(software_denoise_rejects_disoccluded_history)
{
    HeapAllocator allocator;
    SplitScene scene;
    SpatioTemporalDenoiser denoiser = MakeDenoiser(allocator);
    Framebuffer output = MakeImage(allocator, kSize, kSize);
    DenoiserStats stats{};
    for (u32 frame = 0; frame < 4; ++frame)
    {
        CG_REQUIRE(FillSplitScene(scene, allocator, frame, 0.1f));
        CG_REQUIRE(
            denoiser.Denoise(scene.noisy, Guides(scene), DenoiserDesc{}, output, &stats).IsOk());
    }
    // The split moves eight columns left: columns [24, 32) change from near
    // to far.
    CG_REQUIRE(FillSplitScene(scene, allocator, 5, 0.1f, 2.0f, 10.0f, kSize / 2 - 8));
    CG_REQUIRE(denoiser.Denoise(scene.noisy, Guides(scene), DenoiserDesc{}, output, &stats).IsOk());
    CG_EXPECT_EQ(stats.historyRejectedPixels, static_cast<u32>(8 * kSize));
    // The uncovered pixels show the far surface's value, not a ghost of the
    // near one.
    for (i32 y = 4; y < kSize - 4; y += 9)
    {
        CG_EXPECT_TRUE(output.GetPixel(kSize / 2 - 4, y).r > 0.6f);
    }

    // The control: with the depth test effectively off, the same change keeps
    // history and the ghost of the near surface is visible.
    SpatioTemporalDenoiser control = MakeDenoiser(allocator);
    DenoiserDesc lax{};
    lax.depthRejectRelative = 1.0e6f;
    lax.spatial = false;
    for (u32 frame = 0; frame < 4; ++frame)
    {
        CG_REQUIRE(FillSplitScene(scene, allocator, frame, 0.1f));
        CG_REQUIRE(control.Denoise(scene.noisy, Guides(scene), lax, output, &stats).IsOk());
    }
    CG_REQUIRE(FillSplitScene(scene, allocator, 5, 0.1f, 2.0f, 10.0f, kSize / 2 - 8));
    CG_REQUIRE(control.Denoise(scene.noisy, Guides(scene), lax, output, &stats).IsOk());
    CG_EXPECT_EQ(stats.historyRejectedPixels, u32{0});
    CG_EXPECT_TRUE(output.GetPixel(kSize / 2 - 4, kSize / 2).r < 0.6f);
}

// A camera pan: content moves right four pixels and the velocity buffer says
// so. History follows it, and only the column strip entering from the left
// edge has nothing to reproject from.
CG_TEST(software_denoise_follows_velocity)
{
    HeapAllocator allocator;
    SplitScene scene;
    SpatioTemporalDenoiser denoiser = MakeDenoiser(allocator);
    Framebuffer output = MakeImage(allocator, kSize, kSize);
    Framebuffer velocity = MakeImage(allocator, kSize, kSize);
    CG_REQUIRE(velocity.IsValid());
    velocity.Clear(Color{4.0f, 0.0f, 0.0f, 0.0f});
    DenoiserStats stats{};
    CG_REQUIRE(FillSplitScene(scene, allocator, 0, 0.1f, 2.0f, 10.0f, kSize / 2));
    CG_REQUIRE(denoiser.Denoise(scene.noisy, Guides(scene), DenoiserDesc{}, output, &stats).IsOk());
    CG_REQUIRE(FillSplitScene(scene, allocator, 1, 0.1f, 2.0f, 10.0f, kSize / 2 + 4));
    CG_REQUIRE(
        denoiser.Denoise(scene.noisy, Guides(scene, &velocity), DenoiserDesc{}, output, &stats)
            .IsOk());
    CG_EXPECT_EQ(stats.historyRejectedPixels, static_cast<u32>(4 * kSize));

    // Without the velocity buffer the moved edge is a disocclusion strip.
    SpatioTemporalDenoiser still = MakeDenoiser(allocator);
    CG_REQUIRE(FillSplitScene(scene, allocator, 0, 0.1f, 2.0f, 10.0f, kSize / 2));
    CG_REQUIRE(still.Denoise(scene.noisy, Guides(scene), DenoiserDesc{}, output, &stats).IsOk());
    CG_REQUIRE(FillSplitScene(scene, allocator, 1, 0.1f, 2.0f, 10.0f, kSize / 2 + 4));
    CG_REQUIRE(still.Denoise(scene.noisy, Guides(scene), DenoiserDesc{}, output, &stats).IsOk());
    CG_EXPECT_EQ(stats.historyRejectedPixels, static_cast<u32>(4 * kSize));
}

CG_TEST(software_denoise_is_deterministic_across_jobs)
{
    HeapAllocator allocator;
    SplitScene scene;
    CG_REQUIRE(FillSplitScene(scene, allocator, 3, 0.3f));
    SpatioTemporalDenoiser serial = MakeDenoiser(allocator);
    SpatioTemporalDenoiser parallel = MakeDenoiser(allocator);
    Framebuffer a = MakeImage(allocator, kSize, kSize);
    Framebuffer b = MakeImage(allocator, kSize, kSize);
    cyber::jobs::JobSystem::Desc jobDesc{};
    jobDesc.workerCount = 4;
    auto jobs = cyber::jobs::JobSystem::Create(jobDesc, allocator);
    CG_REQUIRE(jobs.IsOk());
    DenoiserDesc desc{};
    for (u32 frame = 0; frame < 3; ++frame)
    {
        CG_REQUIRE(FillSplitScene(scene, allocator, frame, 0.3f));
        desc.useJobs = false;
        CG_REQUIRE(serial.Denoise(scene.noisy, Guides(scene), desc, a).IsOk());
        desc.useJobs = true;
        CG_REQUIRE(
            parallel.Denoise(scene.noisy, Guides(scene), desc, b, nullptr, &jobs.Value()).IsOk());
    }
    jobs.Value().Shutdown();
    u32 differing = 0;
    for (i32 y = 0; y < kSize; ++y)
    {
        for (i32 x = 0; x < kSize; ++x)
        {
            const Color pa = a.GetPixel(x, y);
            const Color pb = b.GetPixel(x, y);
            differing += std::memcmp(&pa, &pb, sizeof(Color)) != 0 ? 1u : 0u;
        }
    }
    CG_EXPECT_EQ(differing, u32{0});
}
