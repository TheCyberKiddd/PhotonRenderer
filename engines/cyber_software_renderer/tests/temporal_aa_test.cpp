#include <cyber/jobs/job_system.h>
#include <cyber/math/mat4.h>
#include <cyber/math/vec3.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/post_process_quality.h>
#include <cyber/software/temporal_aa.h>
#include <cyber/test/test.h>

#include <cmath>
#include <utility>

using namespace cyber::software;
using namespace cyber::memory;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::math::Color;
using cyber::math::Mat4;
using cyber::math::Vec2;
using cyber::math::Vec3;

namespace
{

constexpr i32 kSize = 48;

[[nodiscard]] bool MakeImage(HeapAllocator& heap, Framebuffer& out, i32 width, i32 height)
{
    FramebufferDesc desc{};
    desc.width = width;
    desc.height = height;
    desc.format = PixelFormat::RGBA32F;
    auto created = Framebuffer::Create(desc, heap);
    if (!created)
    {
        return false;
    }
    out = std::move(created).Value();
    return true;
}

[[nodiscard]] bool ImagesEqual(const Framebuffer& lhs, const Framebuffer& rhs)
{
    for (i32 y = 0; y < lhs.Height(); ++y)
    {
        for (i32 x = 0; x < lhs.Width(); ++x)
        {
            const Color a = lhs.GetPixel(x, y);
            const Color b = rhs.GetPixel(x, y);
            if (a.r != b.r || a.g != b.g || a.b != b.b || a.a != b.a)
            {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

CG_TEST(RadicalInverseMatchesItsDefinition)
{
    CG_EXPECT_NEAR(RadicalInverse(1u, 2u), 0.5f, 1.0e-6f);
    CG_EXPECT_NEAR(RadicalInverse(2u, 2u), 0.25f, 1.0e-6f);
    CG_EXPECT_NEAR(RadicalInverse(3u, 2u), 0.75f, 1.0e-6f);
    CG_EXPECT_NEAR(RadicalInverse(1u, 3u), 1.0f / 3.0f, 1.0e-6f);
    CG_EXPECT_NEAR(RadicalInverse(0u, 2u), 0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(RadicalInverse(5u, 1u), 0.0f, 1.0e-6f);
}

CG_TEST(JitterCoversThePixelAndAveragesToItsCenter)
{
    constexpr i32 kPhases = 16;
    f32 sumX = 0.0f;
    f32 sumY = 0.0f;
    f32 minX = 1.0f;
    f32 maxX = -1.0f;
    for (u32 frame = 0; frame < static_cast<u32>(kPhases); ++frame)
    {
        const Vec2 offset = TemporalJitterOffset(frame, kPhases);
        CG_EXPECT_TRUE(offset.x >= -0.5f && offset.x <= 0.5f);
        CG_EXPECT_TRUE(offset.y >= -0.5f && offset.y <= 0.5f);
        sumX += offset.x;
        sumY += offset.y;
        minX = std::min(minX, offset.x);
        maxX = std::max(maxX, offset.x);
    }
    // A sequence that did not average out would bias every frame off center.
    CG_EXPECT_NEAR(sumX / static_cast<f32>(kPhases), 0.0f, 0.05f);
    CG_EXPECT_NEAR(sumY / static_cast<f32>(kPhases), 0.0f, 0.05f);
    // And one that bunched up would leave parts of the pixel unsampled.
    CG_EXPECT_TRUE(maxX - minX > 0.8f);

    // The sequence repeats on its period, which is what makes a capture
    // reproducible.
    CG_EXPECT_NEAR(TemporalJitterOffset(0u, kPhases).x,
                   TemporalJitterOffset(static_cast<u32>(kPhases), kPhases).x, 1.0e-6f);
    // One phase is no jitter at all.
    CG_EXPECT_NEAR(TemporalJitterOffset(3u, 1).x, 0.0f, 1.0e-6f);
}

CG_TEST(ProjectionJitterShiftsTheImageByExactlyOnePixelOffset)
{
    constexpr i32 kWidth = 128;
    constexpr i32 kHeight = 64;
    const Mat4 projection = Mat4::PerspectiveReverseZ(
        1.0471975512f, static_cast<f32>(kWidth) / static_cast<f32>(kHeight), 0.1f, 100.0f);

    const Vec2 offset{0.25f, -0.4f};
    const Mat4 jittered = ApplyJitterToProjection(projection, offset, kWidth, kHeight);

    // A point projected through both matrices must land exactly one offset
    // apart in pixels, with depth untouched.
    for (const Vec3 point :
         {Vec3{0.0f, 0.0f, -5.0f}, Vec3{1.5f, -0.75f, -12.0f}, Vec3{-2.0f, 3.0f, -3.0f}})
    {
        const Vec3 base = projection.TransformPoint(point);
        const Vec3 shifted = jittered.TransformPoint(point);
        const f32 pixelX = (shifted.x - base.x) * 0.5f * static_cast<f32>(kWidth);
        const f32 pixelY = -(shifted.y - base.y) * 0.5f * static_cast<f32>(kHeight);
        CG_EXPECT_NEAR(pixelX, offset.x, 1.0e-3f);
        CG_EXPECT_NEAR(pixelY, offset.y, 1.0e-3f);
        CG_EXPECT_NEAR(shifted.z, base.z, 1.0e-6f);
    }

    // A degenerate target or a poisoned offset leaves the matrix alone.
    const Mat4 unchanged = ApplyJitterToProjection(projection, offset, 0, kHeight);
    CG_EXPECT_NEAR(unchanged.columns[2].x, projection.columns[2].x, 1.0e-9f);
}

CG_TEST(FirstFrameIsPrimedNotAccumulated)
{
    HeapAllocator heap;
    auto created = TemporalAaResolver::Create(kSize, kSize, heap);
    CG_REQUIRE(created.IsOk());
    TemporalAaResolver resolver = std::move(created).Value();
    CG_EXPECT_TRUE(resolver.IsValid());
    CG_EXPECT_FALSE(resolver.HasHistory());

    Framebuffer color;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, color, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, destination, kSize, kSize));
    color.Clear(Color{0.3f, 0.6f, 0.9f, 1.0f});

    TemporalAaStats stats{};
    TemporalAaSettings settings{};
    CG_REQUIRE(resolver.Resolve(color, nullptr, destination, settings, nullptr, &stats).IsOk());
    CG_EXPECT_TRUE(stats.historyPrimed);
    CG_EXPECT_TRUE(resolver.HasHistory());
    CG_EXPECT_TRUE(ImagesEqual(color, destination));

    resolver.Reset();
    CG_EXPECT_FALSE(resolver.HasHistory());
}

CG_TEST(ResolveValidatesItsArguments)
{
    HeapAllocator heap;
    auto created = TemporalAaResolver::Create(kSize, kSize, heap);
    CG_REQUIRE(created.IsOk());
    TemporalAaResolver resolver = std::move(created).Value();

    Framebuffer color;
    Framebuffer destination;
    Framebuffer wrongSize;
    CG_REQUIRE(MakeImage(heap, color, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, destination, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, wrongSize, kSize / 2, kSize));

    TemporalAaSettings settings{};
    CG_EXPECT_FALSE(
        resolver.Resolve(wrongSize, nullptr, destination, settings, nullptr, nullptr).IsOk());
    CG_EXPECT_FALSE(resolver.Resolve(color, nullptr, wrongSize, settings, nullptr, nullptr).IsOk());
    CG_EXPECT_FALSE(
        resolver.Resolve(color, &wrongSize, destination, settings, nullptr, nullptr).IsOk());
    CG_EXPECT_FALSE(resolver.Resolve(color, nullptr, color, settings, nullptr, nullptr).IsOk());

    // A history weight of 1 would never converge, so it is rejected rather
    // than silently clamped.
    settings.historyWeight = 1.0f;
    CG_EXPECT_FALSE(
        resolver.Resolve(color, nullptr, destination, settings, nullptr, nullptr).IsOk());

    CG_EXPECT_FALSE(TemporalAaResolver::Create(0, kSize, heap).IsOk());
}

CG_TEST(StaticSceneConvergesTowardTheAccumulatedAverage)
{
    HeapAllocator heap;
    auto created = TemporalAaResolver::Create(kSize, kSize, heap);
    CG_REQUIRE(created.IsOk());
    TemporalAaResolver resolver = std::move(created).Value();

    Framebuffer color;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, color, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, destination, kSize, kSize));

    TemporalAaSettings settings{};
    settings.historyWeight = 0.9f;
    settings.clampHistory = false;
    settings.luminanceWeighting = false;

    // Two alternating sub-pixel phases of a half-covered pixel: the converged
    // answer is their average, which is the sample the pixel never renders.
    for (i32 frame = 0; frame < 64; ++frame)
    {
        color.Clear((frame % 2) == 0 ? Color{1.0f, 1.0f, 1.0f, 1.0f}
                                     : Color{0.0f, 0.0f, 0.0f, 1.0f});
        CG_REQUIRE(
            resolver.Resolve(color, nullptr, destination, settings, nullptr, nullptr).IsOk());
    }
    CG_EXPECT_NEAR(destination.GetPixel(kSize / 2, kSize / 2).r, 0.5f, 0.1f);
}

CG_TEST(NeighborhoodClampStopsStaleHistoryFromSurviving)
{
    HeapAllocator heap;
    auto created = TemporalAaResolver::Create(kSize, kSize, heap);
    CG_REQUIRE(created.IsOk());
    TemporalAaResolver resolver = std::move(created).Value();

    Framebuffer color;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, color, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, destination, kSize, kSize));

    TemporalAaSettings settings{};
    settings.historyWeight = 0.95f;
    settings.clampHistory = true;
    settings.luminanceWeighting = false;

    // Build a white history, then cut to black. With the clamp on, the
    // neighborhood contains no white, so history cannot carry any through.
    color.Clear(Color{1.0f, 1.0f, 1.0f, 1.0f});
    CG_REQUIRE(resolver.Resolve(color, nullptr, destination, settings, nullptr, nullptr).IsOk());
    CG_REQUIRE(resolver.Resolve(color, nullptr, destination, settings, nullptr, nullptr).IsOk());

    color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    TemporalAaStats stats{};
    CG_REQUIRE(resolver.Resolve(color, nullptr, destination, settings, nullptr, &stats).IsOk());
    CG_EXPECT_TRUE(stats.clampedPixels > 0u);
    CG_EXPECT_NEAR(destination.GetPixel(kSize / 2, kSize / 2).r, 0.0f, 1.0e-4f);

    // The control: the same cut with the clamp off keeps most of the old
    // frame, which is exactly the artifact the clamp exists to prevent.
    resolver.Reset();
    settings.clampHistory = false;
    color.Clear(Color{1.0f, 1.0f, 1.0f, 1.0f});
    CG_REQUIRE(resolver.Resolve(color, nullptr, destination, settings, nullptr, nullptr).IsOk());
    CG_REQUIRE(resolver.Resolve(color, nullptr, destination, settings, nullptr, nullptr).IsOk());
    color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    CG_REQUIRE(resolver.Resolve(color, nullptr, destination, settings, nullptr, nullptr).IsOk());
    CG_EXPECT_TRUE(destination.GetPixel(kSize / 2, kSize / 2).r > 0.9f);
}

CG_TEST(VelocityReprojectionFollowsTheMotionAndRejectsDisocclusions)
{
    HeapAllocator heap;
    auto created = TemporalAaResolver::Create(kSize, kSize, heap);
    CG_REQUIRE(created.IsOk());
    TemporalAaResolver resolver = std::move(created).Value();

    Framebuffer color;
    Framebuffer velocity;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, color, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, velocity, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, destination, kSize, kSize));

    TemporalAaSettings settings{};
    settings.historyWeight = 0.9f;
    settings.clampHistory = false;
    settings.luminanceWeighting = false;
    settings.disocclusionPixels = 64.0f;

    // A vertical bar that moves four pixels right per frame, with a velocity
    // buffer that says so. History sampled along that velocity lands on the
    // same bar, so the bar stays sharp instead of smearing.
    const auto drawBar = [&color](i32 position)
    {
        color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
        color.ClearRect(position, 0, 6, kSize, Color{1.0f, 1.0f, 1.0f, 1.0f});
    };
    velocity.Clear(Color{4.0f, 0.0f, 0.0f, 1.0f});

    drawBar(4);
    CG_REQUIRE(resolver.Resolve(color, &velocity, destination, settings, nullptr, nullptr).IsOk());
    for (i32 frame = 1; frame < 6; ++frame)
    {
        drawBar(4 + frame * 4);
        CG_REQUIRE(
            resolver.Resolve(color, &velocity, destination, settings, nullptr, nullptr).IsOk());
    }

    const i32 barStart = 4 + 5 * 4;
    CG_EXPECT_TRUE(destination.GetPixel(barStart + 2, kSize / 2).r > 0.9f);
    // Four pixels behind the bar is where a trail would be.
    CG_EXPECT_TRUE(destination.GetPixel(barStart - 3, kSize / 2).r < 0.15f);

    // Motion past the threshold is a disocclusion: history is dropped and the
    // frame is used as it is.
    velocity.Clear(Color{400.0f, 0.0f, 0.0f, 1.0f});
    TemporalAaStats stats{};
    drawBar(10);
    CG_REQUIRE(resolver.Resolve(color, &velocity, destination, settings, nullptr, &stats).IsOk());
    CG_EXPECT_EQ(stats.rejectedPixels, static_cast<u32>(kSize) * static_cast<u32>(kSize));
    CG_EXPECT_TRUE(ImagesEqual(color, destination));
}

CG_TEST(FlickerRelaxationKeepsASubPixelFeatureAStrictClampLosesAndStillDropsACut)
{
    HeapAllocator heap;

    Framebuffer color;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, color, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, destination, kSize, kSize));

    // A column that is present in half the frames and absent in the other
    // half: a feature narrower than a pixel, as a jittered sample sees it. Its
    // correct resolved value is the average of the two, 0.5, and no single
    // frame ever contains that value anywhere in the column's neighborhood.
    const auto renderPhase = [&color](bool lit)
    {
        color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
        if (lit)
        {
            color.ClearRect(kSize / 2, 0, 1, kSize, Color{1.0f, 1.0f, 1.0f, 1.0f});
        }
    };

    // A failure inside this helper reports as -1, which no assertion below
    // accepts, so a broken resolve cannot pass as a converged one.
    const auto converge = [&](f32 relaxation) -> f32
    {
        TemporalAaSettings settings{};
        settings.historyWeight = 0.9f;
        settings.clampHistory = true;
        settings.flickerRelaxation = relaxation;

        auto created = TemporalAaResolver::Create(kSize, kSize, heap);
        if (!created)
        {
            return -1.0f;
        }
        TemporalAaResolver resolver = std::move(created).Value();
        for (i32 frame = 0; frame < 48; ++frame)
        {
            renderPhase((frame & 1) == 0);
            if (!resolver.Resolve(color, nullptr, destination, settings, nullptr, nullptr).IsOk())
            {
                return -1.0f;
            }
        }
        return destination.GetPixel(kSize / 2, kSize / 2).r;
    };

    // A strict neighborhood clamp cannot hold a value its neighborhood never
    // contains, so the column settles far below its true coverage.
    const f32 strict = converge(0.0f);
    // Widening the box by the amount the pixel has been oscillating lets the
    // accumulation sit where it belongs.
    const f32 relaxed = converge(2.0f);
    CG_EXPECT_TRUE(strict >= 0.0f);
    CG_EXPECT_TRUE(relaxed >= 0.0f);
    CG_EXPECT_TRUE(strict < 0.3f);
    CG_EXPECT_TRUE(relaxed > 0.4f);
    CG_EXPECT_TRUE(relaxed < 0.6f);

    // The same relaxation must not rescue a cut. Nothing has been oscillating
    // here, so there is no amplitude to widen the box with, and the white
    // history is rejected on the frame the scene turns black.
    TemporalAaSettings settings{};
    settings.historyWeight = 0.95f;
    settings.clampHistory = true;
    settings.flickerRelaxation = 2.0f;

    auto cutCreated = TemporalAaResolver::Create(kSize, kSize, heap);
    CG_REQUIRE(cutCreated.IsOk());
    TemporalAaResolver cutResolver = std::move(cutCreated).Value();
    color.Clear(Color{1.0f, 1.0f, 1.0f, 1.0f});
    for (i32 frame = 0; frame < 8; ++frame)
    {
        CG_REQUIRE(
            cutResolver.Resolve(color, nullptr, destination, settings, nullptr, nullptr).IsOk());
    }
    color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    TemporalAaStats stats{};
    CG_REQUIRE(cutResolver.Resolve(color, nullptr, destination, settings, nullptr, &stats).IsOk());
    CG_EXPECT_NEAR(destination.GetPixel(kSize / 2, kSize / 2).r, 0.0f, 1.0e-4f);
    CG_EXPECT_TRUE(stats.clampedPixels > 0u);
    // A still, unchanging scene has no oscillation, so no pixel was relaxed.
    CG_EXPECT_EQ(stats.relaxedPixels, 0u);
}

CG_TEST(RelaxationIsReportedAndSwitchesOffCleanly)
{
    HeapAllocator heap;

    Framebuffer color;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, color, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, destination, kSize, kSize));

    TemporalAaSettings settings{};
    settings.clampHistory = true;
    settings.flickerRelaxation = 2.0f;

    auto created = TemporalAaResolver::Create(kSize, kSize, heap);
    CG_REQUIRE(created.IsOk());
    TemporalAaResolver resolver = std::move(created).Value();

    // Alternating the whole image is the loudest oscillation there is, so once
    // a few frames have passed every pixel has an amplitude to widen with.
    TemporalAaStats stats{};
    for (i32 frame = 0; frame < 8; ++frame)
    {
        color.Clear((frame & 1) == 0 ? Color{1.0f, 1.0f, 1.0f, 1.0f}
                                     : Color{0.0f, 0.0f, 0.0f, 1.0f});
        CG_REQUIRE(resolver.Resolve(color, nullptr, destination, settings, nullptr, &stats).IsOk());
    }
    CG_EXPECT_EQ(stats.relaxedPixels, static_cast<u32>(kSize) * static_cast<u32>(kSize));

    // Switched off, the count goes to zero on the next frame without needing a
    // reset, because the relaxation is read per frame rather than baked into
    // the history.
    settings.flickerRelaxation = 0.0f;
    color.Clear(Color{1.0f, 1.0f, 1.0f, 1.0f});
    CG_REQUIRE(resolver.Resolve(color, nullptr, destination, settings, nullptr, &stats).IsOk());
    CG_EXPECT_EQ(stats.relaxedPixels, 0u);

    // And a reset clears the record of oscillation along with the history.
    resolver.Reset();
    color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    settings.flickerRelaxation = 2.0f;
    CG_REQUIRE(resolver.Resolve(color, nullptr, destination, settings, nullptr, &stats).IsOk());
    CG_EXPECT_TRUE(stats.historyPrimed);
    color.Clear(Color{1.0f, 1.0f, 1.0f, 1.0f});
    CG_REQUIRE(resolver.Resolve(color, nullptr, destination, settings, nullptr, &stats).IsOk());
    CG_EXPECT_EQ(stats.relaxedPixels, 0u);
}

CG_TEST(DepthRejectionStopsAnOccluderFromTrailingAcrossWhatItUncovers)
{
    HeapAllocator heap;

    constexpr f32 kNear = 0.1f;
    constexpr f32 kFar = 100.0f;
    constexpr f32 kBarDistance = 2.0f;
    constexpr f32 kBackgroundDistance = 20.0f;
    constexpr i32 kBarWidth = 8;
    constexpr i32 kStep = 3;

    // Reverse-Z value for a view-space distance, the inverse of
    // LinearizeReverseZ.
    const auto encodeDepth = [](f32 distance)
    { return (kNear * kFar / distance - kNear) / (kFar - kNear); };

    Framebuffer color;
    Framebuffer velocity;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, color, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, velocity, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, destination, kSize, kSize));

    DepthBufferDesc depthDesc{};
    depthDesc.width = kSize;
    depthDesc.height = kSize;
    auto depthCreated = DepthBuffer::Create(depthDesc, heap);
    CG_REQUIRE(depthCreated.IsOk());
    DepthBuffer depth = std::move(depthCreated).Value();

    // A bright bar sliding right across a static dark background. The
    // background is not moving, so its velocity is zero and every pixel the
    // bar uncovers reprojects onto itself -- where the bar was. That is the
    // disocclusion a velocity buffer cannot describe: the velocity is
    // correct, and the history it points at is still the wrong surface.
    const auto renderFrame = [&](i32 barStart)
    {
        color.Clear(Color{0.05f, 0.05f, 0.05f, 1.0f});
        color.ClearRect(barStart, 0, kBarWidth, kSize, Color{1.0f, 1.0f, 1.0f, 1.0f});
        velocity.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
        depth.Clear();
        for (i32 y = 0; y < kSize; ++y)
        {
            for (i32 x = 0; x < kSize; ++x)
            {
                const bool onBar = x >= barStart && x < barStart + kBarWidth;
                CG_EXPECT_TRUE(depth.TestAndWrite(
                    x, y, encodeDepth(onBar ? kBarDistance : kBackgroundDistance)));
            }
        }
    };

    // The pixel the bar has just uncovered on the trailing edge, after the
    // last frame.
    constexpr i32 kFrames = 6;
    const i32 finalBarStart = 4 + (kFrames - 1) * kStep;
    const i32 uncovered = finalBarStart - 1;

    const auto run = [&](bool supplyDepth, TemporalAaStats& stats) -> f32
    {
        auto created = TemporalAaResolver::Create(kSize, kSize, heap);
        if (!created)
        {
            return -1.0f;
        }
        TemporalAaResolver resolver = std::move(created).Value();

        TemporalAaSettings settings{};
        settings.historyWeight = 0.9f;
        settings.nearPlane = kNear;
        settings.farPlane = kFar;

        for (i32 frame = 0; frame < kFrames; ++frame)
        {
            renderFrame(4 + frame * kStep);
            TemporalAaInputs inputs{};
            inputs.color = &color;
            inputs.velocity = &velocity;
            inputs.depth = supplyDepth ? &depth : nullptr;
            if (!resolver.Resolve(inputs, destination, settings, nullptr, &stats).IsOk())
            {
                return -1.0f;
            }
        }
        return destination.GetPixel(uncovered, kSize / 2).r;
    };

    TemporalAaStats withoutDepth{};
    const f32 trailing = run(false, withoutDepth);
    CG_EXPECT_TRUE(trailing >= 0.0f);
    CG_EXPECT_EQ(withoutDepth.depthRejectedPixels, 0u);
    // Velocity alone cannot see the disocclusion, so the bar's white is still
    // sitting on the background it uncovered. This is the artifact.
    CG_EXPECT_TRUE(trailing > 0.3f);

    TemporalAaStats withDepth{};
    const f32 clean = run(true, withDepth);
    CG_EXPECT_TRUE(clean >= 0.0f);
    // The depth test sees that the surface at this pixel is twenty units away
    // and its history was two, drops the history, and the background stays
    // the background.
    CG_EXPECT_TRUE(withDepth.depthRejectedPixels > 0u);
    CG_EXPECT_TRUE(clean < 0.15f);
    CG_EXPECT_TRUE(clean < trailing);

    // A still frame disoccludes nothing, so supplying depth costs no history.
    TemporalAaStats still{};
    auto created = TemporalAaResolver::Create(kSize, kSize, heap);
    CG_REQUIRE(created.IsOk());
    TemporalAaResolver resolver = std::move(created).Value();
    TemporalAaSettings settings{};
    settings.nearPlane = kNear;
    settings.farPlane = kFar;
    for (i32 frame = 0; frame < 4; ++frame)
    {
        renderFrame(10);
        velocity.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
        TemporalAaInputs inputs{};
        inputs.color = &color;
        inputs.velocity = &velocity;
        inputs.depth = &depth;
        CG_REQUIRE(resolver.Resolve(inputs, destination, settings, nullptr, &still).IsOk());
    }
    CG_EXPECT_EQ(still.depthRejectedPixels, 0u);
    CG_EXPECT_EQ(still.rejectedPixels, 0u);
}

CG_TEST(DepthInputIsValidatedAndTheTestStartsOnceItHasSomethingToCompare)
{
    HeapAllocator heap;

    Framebuffer color;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, color, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, destination, kSize, kSize));
    color.Clear(Color{0.4f, 0.4f, 0.4f, 1.0f});

    DepthBufferDesc depthDesc{};
    depthDesc.width = kSize;
    depthDesc.height = kSize;
    auto depthCreated = DepthBuffer::Create(depthDesc, heap);
    CG_REQUIRE(depthCreated.IsOk());
    DepthBuffer depth = std::move(depthCreated).Value();

    DepthBufferDesc smallDesc{};
    smallDesc.width = kSize / 2;
    smallDesc.height = kSize;
    auto smallCreated = DepthBuffer::Create(smallDesc, heap);
    CG_REQUIRE(smallCreated.IsOk());
    DepthBuffer small = std::move(smallCreated).Value();

    auto created = TemporalAaResolver::Create(kSize, kSize, heap);
    CG_REQUIRE(created.IsOk());
    TemporalAaResolver resolver = std::move(created).Value();

    const TemporalAaSettings settings{};

    TemporalAaInputs mismatched{};
    mismatched.color = &color;
    mismatched.depth = &small;
    CG_EXPECT_FALSE(resolver.Resolve(mismatched, destination, settings, nullptr, nullptr).IsOk());

    TemporalAaInputs noColor{};
    CG_EXPECT_FALSE(resolver.Resolve(noColor, destination, settings, nullptr, nullptr).IsOk());

    // The first frame with depth has nothing to compare against and must not
    // reject on it; the depth is recorded for the frame after.
    TemporalAaInputs inputs{};
    inputs.color = &color;
    inputs.depth = &depth;
    TemporalAaStats stats{};
    CG_REQUIRE(resolver.Resolve(inputs, destination, settings, nullptr, &stats).IsOk());
    CG_EXPECT_TRUE(stats.historyPrimed);
    CG_EXPECT_EQ(stats.depthRejectedPixels, 0u);
    CG_REQUIRE(resolver.Resolve(inputs, destination, settings, nullptr, &stats).IsOk());
    CG_EXPECT_FALSE(stats.historyPrimed);
    // The same unchanging depth rejects nothing.
    CG_EXPECT_EQ(stats.depthRejectedPixels, 0u);
}
CG_TEST(TemporalResolveIsDeterministicWithAndWithoutJobs)
{
    HeapAllocator heap;
    auto jobsCreated =
        cyber::jobs::JobSystem::Create(cyber::jobs::JobSystem::Desc{4, nullptr}, heap);
    CG_REQUIRE(jobsCreated.IsOk());
    cyber::jobs::JobSystem jobSystem = std::move(jobsCreated).Value();

    Framebuffer color;
    Framebuffer velocity;
    Framebuffer serial;
    Framebuffer parallel;
    CG_REQUIRE(MakeImage(heap, color, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, velocity, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, serial, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, parallel, kSize, kSize));

    auto serialCreated = TemporalAaResolver::Create(kSize, kSize, heap);
    CG_REQUIRE(serialCreated.IsOk());
    TemporalAaResolver serialResolver = std::move(serialCreated).Value();
    auto parallelCreated = TemporalAaResolver::Create(kSize, kSize, heap);
    CG_REQUIRE(parallelCreated.IsOk());
    TemporalAaResolver parallelResolver = std::move(parallelCreated).Value();

    TemporalAaSettings settings{};
    for (i32 frame = 0; frame < 5; ++frame)
    {
        for (i32 y = 0; y < kSize; ++y)
        {
            for (i32 x = 0; x < kSize; ++x)
            {
                const f32 value = 0.5f + 0.5f * std::sin(static_cast<f32>(x + frame) * 0.4f) *
                                             std::cos(static_cast<f32>(y) * 0.27f);
                color.SetPixel(x, y, Color{value * 2.0f, value, value * 0.5f, 1.0f});
                velocity.SetPixel(x, y, Color{1.5f, -0.5f, 0.0f, 1.0f});
            }
        }

        settings.useJobs = false;
        CG_REQUIRE(
            serialResolver.Resolve(color, &velocity, serial, settings, nullptr, nullptr).IsOk());
        settings.useJobs = true;
        CG_REQUIRE(
            parallelResolver.Resolve(color, &velocity, parallel, settings, &jobSystem, nullptr)
                .IsOk());
        CG_EXPECT_TRUE(ImagesEqual(serial, parallel));
    }
}

CG_TEST(FxaaSoftensEdgesAndLeavesFlatAreasAlone)
{
    HeapAllocator heap;
    Framebuffer source;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, source, 64, 64));
    CG_REQUIRE(MakeImage(heap, destination, 64, 64));

    // A near-vertical edge, the aliasing case an edge filter exists for.
    for (i32 y = 0; y < 64; ++y)
    {
        for (i32 x = 0; x < 64; ++x)
        {
            const bool inside = static_cast<f32>(x) < 20.0f + static_cast<f32>(y) * 0.2f;
            source.SetPixel(
                x, y, inside ? Color{1.0f, 1.0f, 1.0f, 1.0f} : Color{0.05f, 0.05f, 0.05f, 1.0f});
        }
    }

    FxaaSettings settings{};
    CG_REQUIRE(FxaaResolve(source, destination, settings, nullptr).IsOk());

    EdgeAliasingMetrics before{};
    EdgeAliasingMetrics after{};
    CG_REQUIRE(MeasureEdgeAliasing(source, 0.5f, before).IsOk());
    CG_REQUIRE(MeasureEdgeAliasing(destination, 0.5f, after).IsOk());
    CG_EXPECT_TRUE(after.hardStepCount < before.hardStepCount);

    // A flat region must come through untouched, or the filter is just a blur.
    CG_EXPECT_NEAR(destination.GetPixel(60, 10).r, 0.05f, 1.0e-5f);
    CG_EXPECT_NEAR(destination.GetPixel(2, 40).r, 1.0f, 1.0e-5f);

    CG_EXPECT_FALSE(FxaaResolve(source, source, settings, nullptr).IsOk());
}
