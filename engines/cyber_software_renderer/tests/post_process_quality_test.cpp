// Unit tests for the Phase 30 image measurements.
//
// The acceptance suite asserts thresholds on the numbers these functions
// return, so the functions themselves need cases whose answers are known
// before they are run. Every image here is built so the expected result can be
// written down by hand.

#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/post_process_quality.h>
#include <cyber/test/test.h>

#include <utility>

using namespace cyber::software;
using namespace cyber::memory;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::math::Color;

namespace
{

constexpr i32 kSize = 16;

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

}  // namespace

CG_TEST(ImageDifferenceCountsEveryChannelAndPixel)
{
    HeapAllocator heap;
    Framebuffer image;
    Framebuffer reference;
    CG_REQUIRE(MakeImage(heap, image, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, reference, kSize, kSize));

    reference.Clear(Color{0.5f, 0.5f, 0.5f, 1.0f});
    image.Clear(Color{0.5f, 0.5f, 0.5f, 1.0f});

    ImageDifferenceMetrics identical{};
    CG_REQUIRE(MeasureImageDifference(image, reference, 1.0f / 255.0f, identical).IsOk());
    CG_EXPECT_EQ(identical.pixelCount, static_cast<u32>(kSize * kSize));
    CG_EXPECT_NEAR(identical.meanAbsoluteError, 0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(identical.maxAbsoluteError, 0.0f, 1.0e-6f);
    CG_EXPECT_EQ(identical.differingPixels, 0u);

    // One row of sixteen, off by a quarter on one channel. The mean is over
    // every channel of every pixel: 0.25 on one of three channels, on one of
    // sixteen rows.
    for (i32 x = 0; x < kSize; ++x)
    {
        image.SetPixel(x, 0, Color{0.75f, 0.5f, 0.5f, 1.0f});
    }

    ImageDifferenceMetrics difference{};
    CG_REQUIRE(MeasureImageDifference(image, reference, 1.0f / 255.0f, difference).IsOk());
    CG_EXPECT_NEAR(difference.maxAbsoluteError, 0.25f, 1.0e-6f);
    CG_EXPECT_NEAR(difference.meanAbsoluteError, 0.25f / 3.0f / 16.0f, 1.0e-6f);
    CG_EXPECT_EQ(difference.differingPixels, static_cast<u32>(kSize));
    CG_EXPECT_NEAR(difference.differingFraction, 1.0f / 16.0f, 1.0e-6f);
    // A few large errors weigh more in the root mean square than in the mean.
    CG_EXPECT_TRUE(difference.rootMeanSquareError > difference.meanAbsoluteError);
}

CG_TEST(ImageDifferenceRejectsMismatchedInput)
{
    HeapAllocator heap;
    Framebuffer image;
    Framebuffer small;
    Framebuffer invalid;
    CG_REQUIRE(MakeImage(heap, image, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, small, kSize / 2, kSize));

    ImageDifferenceMetrics metrics{};
    CG_EXPECT_FALSE(MeasureImageDifference(image, small, 1.0f / 255.0f, metrics).IsOk());
    CG_EXPECT_FALSE(MeasureImageDifference(image, invalid, 1.0f / 255.0f, metrics).IsOk());
    CG_EXPECT_FALSE(MeasureImageDifference(image, image, -1.0f, metrics).IsOk());
}

CG_TEST(GhostingReadsAsTheFractionOfTheStaleFrameThatSurvived)
{
    HeapAllocator heap;
    Framebuffer resolved;
    Framebuffer current;
    Framebuffer previous;
    CG_REQUIRE(MakeImage(heap, resolved, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, current, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, previous, kSize, kSize));

    // The two references disagree everywhere by a full unit of luminance, so
    // every pixel is a changed pixel and the axis length is known.
    current.Clear(Color{1.0f, 1.0f, 1.0f, 1.0f});
    previous.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});

    // Landing exactly on the correct answer is zero contamination.
    resolved.Clear(Color{1.0f, 1.0f, 1.0f, 1.0f});
    GhostingMetrics clean{};
    CG_REQUIRE(MeasureGhosting(resolved, current, previous, 0.05f, 0.25f, clean).IsOk());
    CG_EXPECT_EQ(clean.changedPixels, static_cast<u32>(kSize * kSize));
    CG_EXPECT_NEAR(clean.changedFraction, 1.0f, 1.0e-6f);
    CG_EXPECT_NEAR(clean.meanContamination, 0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(clean.weightedContamination, 0.0f, 1.0e-6f);
    CG_EXPECT_EQ(clean.ghostedPixels, 0u);

    // Reproducing the stale answer is complete contamination.
    resolved.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    GhostingMetrics stale{};
    CG_REQUIRE(MeasureGhosting(resolved, current, previous, 0.05f, 0.25f, stale).IsOk());
    CG_EXPECT_NEAR(stale.meanContamination, 1.0f, 1.0e-6f);
    CG_EXPECT_NEAR(stale.weightedContamination, 1.0f, 1.0e-6f);
    CG_EXPECT_NEAR(stale.maxContamination, 1.0f, 1.0e-6f);
    CG_EXPECT_EQ(stale.ghostedPixels, static_cast<u32>(kSize * kSize));

    // A quarter of the way back toward the stale answer reads as a quarter,
    // and lands under the ghosting threshold.
    resolved.Clear(Color{0.75f, 0.75f, 0.75f, 1.0f});
    GhostingMetrics quarter{};
    CG_REQUIRE(MeasureGhosting(resolved, current, previous, 0.05f, 0.25f, quarter).IsOk());
    CG_EXPECT_NEAR(quarter.meanContamination, 0.25f, 1.0e-6f);
    CG_EXPECT_NEAR(quarter.weightedContamination, 0.25f, 1.0e-6f);
    CG_EXPECT_EQ(quarter.ghostedPixels, 0u);
}

CG_TEST(WeightedGhostingIgnoresPixelsThatBarelyMoved)
{
    HeapAllocator heap;
    Framebuffer resolved;
    Framebuffer current;
    Framebuffer previous;
    CG_REQUIRE(MakeImage(heap, resolved, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, current, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, previous, kSize, kSize));

    // This is the case the two numbers are meant to separate. Every pixel
    // moved, but the top row moved by a full unit and the rest moved by barely
    // more than the change threshold. The resolve is perfect on the row that
    // carries the image and fully stale on the rows that carry almost nothing.
    //
    // The per-pixel mean sees fifteen of sixteen rows fully contaminated and
    // calls the image a smear. The fitted number weighs each pixel by how far
    // it actually moved, and reports that almost none of the previous frame
    // survived, which is what the image looks like.
    constexpr f32 kTiny = 0.04f;
    for (i32 y = 0; y < kSize; ++y)
    {
        const bool loud = y == 0;
        const f32 offset = loud ? 1.0f : kTiny;
        for (i32 x = 0; x < kSize; ++x)
        {
            current.SetPixel(x, y, Color{0.5f, 0.5f, 0.5f, 1.0f});
            previous.SetPixel(x, y, Color{0.5f + offset, 0.5f + offset, 0.5f + offset, 1.0f});
            // Perfect on the loud row, entirely stale on the quiet ones.
            resolved.SetPixel(x, y,
                              loud ? Color{0.5f, 0.5f, 0.5f, 1.0f}
                                   : Color{0.5f + offset, 0.5f + offset, 0.5f + offset, 1.0f});
        }
    }

    GhostingMetrics metrics{};
    CG_REQUIRE(MeasureGhosting(resolved, current, previous, 0.05f, 0.25f, metrics).IsOk());
    CG_EXPECT_EQ(metrics.changedPixels, static_cast<u32>(kSize * kSize));
    // Fifteen of sixteen rows read as fully stale.
    CG_EXPECT_TRUE(metrics.meanContamination > 0.9f);
    // The energy is in the row that moved, and that row is clean.
    CG_EXPECT_TRUE(metrics.weightedContamination < 0.1f);
}

CG_TEST(GhostingRefusesAStillSceneAndBadArguments)
{
    HeapAllocator heap;
    Framebuffer resolved;
    Framebuffer current;
    Framebuffer previous;
    Framebuffer small;
    CG_REQUIRE(MakeImage(heap, resolved, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, current, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, previous, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, small, kSize, kSize / 2));

    const Color grey{0.5f, 0.5f, 0.5f, 1.0f};
    resolved.Clear(grey);
    current.Clear(grey);
    previous.Clear(grey);

    GhostingMetrics metrics{};
    // A scene that did not move cannot show ghosting, and saying so is not the
    // same as reporting zero.
    CG_EXPECT_FALSE(MeasureGhosting(resolved, current, previous, 0.05f, 0.25f, metrics).IsOk());
    CG_EXPECT_FALSE(MeasureGhosting(resolved, current, small, 0.05f, 0.25f, metrics).IsOk());
    CG_EXPECT_FALSE(MeasureGhosting(resolved, current, previous, 0.0f, 0.25f, metrics).IsOk());
    CG_EXPECT_FALSE(MeasureGhosting(resolved, current, previous, 0.05f, -1.0f, metrics).IsOk());
}

CG_TEST(TemporalStabilitySeparatesAFrozenSequenceFromAFlickeringOne)
{
    HeapAllocator heap;
    Framebuffer frames[4];
    for (i32 index = 0; index < 4; ++index)
    {
        CG_REQUIRE(MakeImage(heap, frames[index], kSize, kSize));
        frames[index].Clear(Color{0.5f, 0.5f, 0.5f, 1.0f});
    }

    const Framebuffer* sequence[4] = {&frames[0], &frames[1], &frames[2], &frames[3]};
    TemporalStabilityMetrics frozen{};
    CG_REQUIRE(
        MeasureTemporalStability(Span<const Framebuffer* const>{sequence, 4}, frozen).IsOk());
    CG_EXPECT_EQ(frozen.frameCount, 4u);
    CG_EXPECT_NEAR(frozen.meanFrameDelta, 0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(frozen.totalDrift, 0.0f, 1.0e-6f);

    // Alternating by a fifth each frame: every consecutive delta is 0.2, and
    // the first and last frame happen to agree, which is exactly the drift a
    // per-frame delta would miss and this one reports as zero.
    frames[1].Clear(Color{0.7f, 0.7f, 0.7f, 1.0f});
    frames[3].Clear(Color{0.7f, 0.7f, 0.7f, 1.0f});
    TemporalStabilityMetrics flickering{};
    CG_REQUIRE(
        MeasureTemporalStability(Span<const Framebuffer* const>{sequence, 4}, flickering).IsOk());
    CG_EXPECT_NEAR(flickering.meanFrameDelta, 0.2f, 1.0e-5f);
    CG_EXPECT_NEAR(flickering.maxFrameDelta, 0.2f, 1.0e-5f);
    CG_EXPECT_NEAR(flickering.totalDrift, 0.2f, 1.0e-5f);

    // A drift the per-frame deltas hide: every step is small, but the sequence
    // walks away from where it started.
    for (i32 index = 0; index < 4; ++index)
    {
        const f32 value = 0.5f + 0.02f * static_cast<f32>(index);
        frames[index].Clear(Color{value, value, value, 1.0f});
    }
    TemporalStabilityMetrics drifting{};
    CG_REQUIRE(
        MeasureTemporalStability(Span<const Framebuffer* const>{sequence, 4}, drifting).IsOk());
    CG_EXPECT_NEAR(drifting.maxFrameDelta, 0.02f, 1.0e-5f);
    CG_EXPECT_NEAR(drifting.totalDrift, 0.06f, 1.0e-5f);
    CG_EXPECT_TRUE(drifting.totalDrift > drifting.maxFrameDelta);

    TemporalStabilityMetrics rejected{};
    CG_EXPECT_FALSE(
        MeasureTemporalStability(Span<const Framebuffer* const>{sequence, 1}, rejected).IsOk());
}

CG_TEST(EdgeAliasingCountsStepsAndNotSoftness)
{
    HeapAllocator heap;
    Framebuffer stepped;
    Framebuffer ramped;
    Framebuffer flat;
    CG_REQUIRE(MakeImage(heap, stepped, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, ramped, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, flat, kSize, kSize));

    // A hard vertical edge down the middle, the same edge spread over two
    // pixels, and nothing at all.
    for (i32 y = 0; y < kSize; ++y)
    {
        for (i32 x = 0; x < kSize; ++x)
        {
            const f32 hard = x < kSize / 2 ? 0.0f : 1.0f;
            stepped.SetPixel(x, y, Color{hard, hard, hard, 1.0f});

            f32 soft = hard;
            if (x == kSize / 2 - 1)
            {
                soft = 1.0f / 3.0f;
            }
            else if (x == kSize / 2)
            {
                soft = 2.0f / 3.0f;
            }
            ramped.SetPixel(x, y, Color{soft, soft, soft, 1.0f});
            flat.SetPixel(x, y, Color{0.5f, 0.5f, 0.5f, 1.0f});
        }
    }

    EdgeAliasingMetrics hard{};
    EdgeAliasingMetrics soft{};
    EdgeAliasingMetrics none{};
    CG_REQUIRE(MeasureEdgeAliasing(stepped, 0.5f, hard).IsOk());
    CG_REQUIRE(MeasureEdgeAliasing(ramped, 0.5f, soft).IsOk());
    CG_REQUIRE(MeasureEdgeAliasing(flat, 0.5f, none).IsOk());

    CG_EXPECT_EQ(hard.pixelCount, static_cast<u32>(kSize * kSize));
    CG_EXPECT_TRUE(hard.hardStepCount > 0u);
    // Spreading one step of 1.0 over two steps of a third each puts both under
    // a threshold of a half, so the staircase is gone while the edge is not.
    CG_EXPECT_EQ(soft.hardStepCount, 0u);
    CG_EXPECT_EQ(none.hardStepCount, 0u);
    // The mean gradient separates a softened edge from a flat image, which a
    // step count on its own cannot.
    CG_EXPECT_TRUE(soft.meanGradient > none.meanGradient);
    CG_EXPECT_TRUE(hard.meanGradient > 0.0f);

    EdgeAliasingMetrics rejected{};
    CG_EXPECT_FALSE(MeasureEdgeAliasing(stepped, 0.0f, rejected).IsOk());
}
