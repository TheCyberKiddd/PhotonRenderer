// Phase 30 acceptance suite.
//
// The roadmap's bar for this phase is two claims: every effect toggles at
// runtime with its time shown, and temporal anti-aliasing shows no visible
// ghosting in a moving test scene. Both are measured here rather than
// inspected, and each threshold is paired with a control configuration that
// fails it, so a measurement that nothing can fail does not pass as evidence.
//
// The test scene is analytic: a scrolling checkerboard with a bank of thin
// vertical lines, evaluated as a function of position. That makes three things
// available exactly, which a rasterized scene could only approximate:
//
//   - the ground truth for any frame, by supersampling the function 8x8;
//   - a one-sample-per-pixel jittered render, the input temporal
//     anti-aliasing is meant to accumulate;
//   - the velocity buffer, which is the scroll rate, with no reprojection
//     error of its own to hide behind.

#include <cyber/jobs/job_system.h>
#include <cyber/math/mat4.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/post_process.h>
#include <cyber/software/post_process_quality.h>
#include <cyber/software/temporal_aa.h>
#include <cyber/test/test.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

using namespace cyber::software;
using namespace cyber::memory;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::math::Color;
using cyber::math::Mat4;
using cyber::math::Vec2;

namespace
{

constexpr i32 kWidth = 96;
constexpr i32 kHeight = 72;
constexpr i32 kReferenceFactor = 8;
constexpr f32 kScrollPixelsPerFrame = 3.0f;

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

[[nodiscard]] bool MakeDepth(HeapAllocator& heap, DepthBuffer& out, i32 width, i32 height)
{
    DepthBufferDesc desc{};
    desc.width = width;
    desc.height = height;
    auto created = DepthBuffer::Create(desc, heap);
    if (!created)
    {
        return false;
    }
    out = std::move(created).Value();
    return true;
}

// The scene, as a function of continuous screen position. x is in scrolled
// coordinates, so a frame is the same function sampled at an offset.
[[nodiscard]] Color SceneColor(f32 x, f32 y, bool thinLines = true) noexcept
{
    // Thin vertical lines: a half-pixel-wide feature every nine pixels, the
    // case that aliases worst and ghosts most visibly.
    const f32 linePhase = x - 9.0f * std::floor(x / 9.0f);
    if (thinLines && linePhase < 0.5f && y > 8.0f && y < static_cast<f32>(kHeight) - 8.0f)
    {
        return Color{1.0f, 0.95f, 0.8f, 1.0f};
    }

    // A checkerboard underneath, so there is large-scale structure for a
    // reprojection error to displace.
    const i32 checkerX = static_cast<i32>(std::floor(x / 12.0f));
    const i32 checkerY = static_cast<i32>(std::floor(y / 12.0f));
    const bool dark = ((checkerX + checkerY) & 1) == 0;
    return dark ? Color{0.06f, 0.07f, 0.09f, 1.0f} : Color{0.55f, 0.58f, 0.62f, 1.0f};
}

// One sample per pixel at the frame's jitter offset: what the renderer
// actually produces for a temporally anti-aliased frame.
void RenderJittered(Framebuffer& out, f32 scroll, const Vec2& jitter, bool thinLines = true) noexcept
{
    for (i32 y = 0; y < out.Height(); ++y)
    {
        for (i32 x = 0; x < out.Width(); ++x)
        {
            const f32 sampleX = static_cast<f32>(x) + 0.5f + jitter.x + scroll;
            const f32 sampleY = static_cast<f32>(y) + 0.5f + jitter.y;
            out.SetPixel(x, y, SceneColor(sampleX, sampleY, thinLines));
        }
    }
}

// Ground truth: the same frame integrated over each pixel footprint.
void RenderReference(Framebuffer& out, f32 scroll, bool thinLines = true) noexcept
{
    const f32 step = 1.0f / static_cast<f32>(kReferenceFactor);
    for (i32 y = 0; y < out.Height(); ++y)
    {
        for (i32 x = 0; x < out.Width(); ++x)
        {
            Color sum{0.0f, 0.0f, 0.0f, 0.0f};
            for (i32 subY = 0; subY < kReferenceFactor; ++subY)
            {
                for (i32 subX = 0; subX < kReferenceFactor; ++subX)
                {
                    const f32 sampleX = static_cast<f32>(x) +
                                        (static_cast<f32>(subX) + 0.5f) * step + scroll;
                    const f32 sampleY =
                        static_cast<f32>(y) + (static_cast<f32>(subY) + 0.5f) * step;
                    const Color sample = SceneColor(sampleX, sampleY, thinLines);
                    sum = Color{sum.r + sample.r, sum.g + sample.g, sum.b + sample.b, 1.0f};
                }
            }
            const f32 inverse =
                1.0f / static_cast<f32>(kReferenceFactor * kReferenceFactor);
            out.SetPixel(x, y, Color{sum.r * inverse, sum.g * inverse, sum.b * inverse, 1.0f});
        }
    }
}

struct GhostingRun
{
    GhostingMetrics metrics{};
    bool ok = false;
};

// Runs frameCount frames of the scrolling scene through the resolver and
// measures the last one against the ground truth of that frame and the one
// before it.
[[nodiscard]] GhostingRun RunScrollingScene(HeapAllocator& heap, const TemporalAaSettings& settings,
                                            bool supplyVelocity, i32 frameCount,
                                            bool thinLines = true)
{
    GhostingRun run{};

    Framebuffer jittered;
    Framebuffer velocity;
    Framebuffer resolved;
    Framebuffer current;
    Framebuffer previous;
    if (!MakeImage(heap, jittered, kWidth, kHeight) || !MakeImage(heap, velocity, kWidth, kHeight) ||
        !MakeImage(heap, resolved, kWidth, kHeight) || !MakeImage(heap, current, kWidth, kHeight) ||
        !MakeImage(heap, previous, kWidth, kHeight))
    {
        return run;
    }
    // Raising the scroll offset samples the scene further right, so a feature
    // moves left on screen: the screen-space velocity is negative, and history
    // is read at pixel minus that velocity.
    velocity.Clear(Color{-kScrollPixelsPerFrame, 0.0f, 0.0f, 1.0f});

    auto created = TemporalAaResolver::Create(kWidth, kHeight, heap);
    if (!created)
    {
        return run;
    }
    TemporalAaResolver resolver = std::move(created).Value();

    for (i32 frame = 0; frame < frameCount; ++frame)
    {
        const f32 scroll = static_cast<f32>(frame) * kScrollPixelsPerFrame;
        RenderJittered(jittered, scroll, TemporalJitterOffset(static_cast<u32>(frame)), thinLines);
        if (!resolver
                 .Resolve(jittered, supplyVelocity ? &velocity : nullptr, resolved, settings,
                          nullptr, nullptr)
                 .IsOk())
        {
            return run;
        }
    }

    RenderReference(current, static_cast<f32>(frameCount - 1) * kScrollPixelsPerFrame, thinLines);
    RenderReference(previous, static_cast<f32>(frameCount - 2) * kScrollPixelsPerFrame, thinLines);

    // changeThreshold keeps pixels the motion did not touch out of the
    // measurement; ghostThreshold is where contamination becomes a trail.
    run.ok = MeasureGhosting(resolved, current, previous, 0.05f, 0.25f, run.metrics).IsOk();
    return run;
}

}  // namespace

// ---------------------------------------------------------------------------
// Done-when, part one: every effect toggles at runtime with its time shown.
// ---------------------------------------------------------------------------

CG_TEST(EveryEffectTogglesAtRuntimeAndReportsItsTime)
{
    constexpr f32 kNear = 0.1f;
    constexpr f32 kFar = 100.0f;

    HeapAllocator heap;
    auto stackCreated = PostProcessStack::Create(kWidth, kHeight, heap);
    CG_REQUIRE(stackCreated.IsOk());
    PostProcessStack stack = std::move(stackCreated).Value();

    Framebuffer color;
    Framebuffer normals;
    Framebuffer velocity;
    Framebuffer destination;
    Framebuffer reference;
    DepthBuffer depth;
    CG_REQUIRE(MakeImage(heap, color, kWidth, kHeight));
    CG_REQUIRE(MakeImage(heap, normals, kWidth, kHeight));
    CG_REQUIRE(MakeImage(heap, velocity, kWidth, kHeight));
    CG_REQUIRE(MakeImage(heap, destination, kWidth, kHeight));
    CG_REQUIRE(MakeImage(heap, reference, kWidth, kHeight));
    CG_REQUIRE(MakeDepth(heap, depth, kWidth, kHeight));

    // A scene with something for every pass to act on: a depth step for
    // occlusion, a distant half for defocus, motion for the blur, and a
    // highlight well above the tone mapper's shoulder for bloom.
    for (i32 y = 0; y < kHeight; ++y)
    {
        for (i32 x = 0; x < kWidth; ++x)
        {
            const Color scene = SceneColor(static_cast<f32>(x) + 0.5f, static_cast<f32>(y) + 0.5f);
            color.SetPixel(x, y, scene);
            normals.SetPixel(x, y, Color{0.5f, 0.5f, 0.0f, 1.0f});
            velocity.SetPixel(x, y, Color{6.0f, 0.0f, 0.0f, 1.0f});
            // Three depth bands: two near ones a crease apart, which is what
            // ambient occlusion acts on, and a distant one for the lens to
            // defocus. A step of tens of units is a different surface, not a
            // crease, and correctly produces no occlusion.
            const f32 distance = x < kWidth / 3 ? 4.0f : (x < 2 * kWidth / 3 ? 4.6f : 30.0f);
            depth.TestAndWrite(x, y, (kNear * kFar / distance - kNear) / (kFar - kNear));
        }
    }
    color.ClearRect(kWidth / 2 - 6, kHeight / 2 - 6, 12, 12, Color{40.0f, 38.0f, 30.0f, 1.0f});

    PostProcessInputs inputs{};
    inputs.color = &color;
    inputs.depth = &depth;
    inputs.normals = &normals;
    inputs.velocity = &velocity;

    auto lutCreated = ColorGradeLut::Bake(
        [] {
            ColorGradeSettings settings{};
            settings.contrast = 1.25f;
            settings.saturation = 0.85f;
            settings.temperature = 0.2f;
            return settings;
        }(),
        kDefaultColorGradeLutSize, heap);
    CG_REQUIRE(lutCreated.IsOk());
    const ColorGradeLut lut = std::move(lutCreated).Value();

    PostProcessSettings all{};
    all.enableAmbientOcclusion = true;
    all.enableDepthOfField = true;
    all.enableMotionBlur = true;
    all.enableBloom = true;
    all.enableExposure = true;
    all.enableToneMap = true;
    all.enableColorGrade = true;
    all.colorGradeLut = &lut;
    all.depthOfField.focusDistance = 4.0f;
    all.depthOfField.apertureFStop = 1.8f;
    all.ambientOcclusion.radius = 1.0f;

    PostProcessView view{};
    view.projection = Mat4::PerspectiveReverseZ(
        1.0471975512f, static_cast<f32>(kWidth) / static_cast<f32>(kHeight), kNear, kFar);
    view.nearPlane = kNear;
    view.farPlane = kFar;

    PostProcessTimings timings{};
    CG_REQUIRE(
        stack.Execute(inputs, reference, all, view, 1.0f / 60.0f, nullptr, &timings).IsOk());

    cyber::core::u64 summed = 0;
    for (const PassTiming& timing : timings.passes)
    {
        CG_EXPECT_TRUE(timing.name != nullptr);
        CG_EXPECT_TRUE(timing.executed);
        // A pass that reports no time is a pass with no readout, which is half
        // of what this phase promises.
        CG_EXPECT_TRUE(timing.nanoseconds > 0u);
        summed += timing.nanoseconds;
        std::printf("[phase30] pass %-18s %8.3f ms\n", timing.name,
                    static_cast<double>(timing.nanoseconds) * 1.0e-6);
    }
    std::printf("[phase30] chain total %8.3f ms, exposure %+.3f EV, average luminance %.4f\n",
                static_cast<double>(timings.totalNanoseconds) * 1.0e-6,
                static_cast<double>(timings.exposureEv),
                static_cast<double>(timings.averageLuminance));
    CG_EXPECT_TRUE(timings.totalNanoseconds > 0u);
    // The parts cannot cost more than the whole.
    CG_EXPECT_TRUE(summed <= timings.totalNanoseconds);

    // Each toggle, one at a time: the pass stops reporting, and the image it
    // was producing changes. A toggle that changed nothing would be a
    // checkbox, not an effect.
    struct Toggle
    {
        PostProcessPass pass;
        bool PostProcessSettings::*flag;
    };
    const Toggle toggles[] = {
        {PostProcessPass::AmbientOcclusion, &PostProcessSettings::enableAmbientOcclusion},
        {PostProcessPass::DepthOfField, &PostProcessSettings::enableDepthOfField},
        {PostProcessPass::MotionBlur, &PostProcessSettings::enableMotionBlur},
        {PostProcessPass::Bloom, &PostProcessSettings::enableBloom},
        {PostProcessPass::ToneMap, &PostProcessSettings::enableToneMap},
        {PostProcessPass::ColorGrade, &PostProcessSettings::enableColorGrade},
    };

    for (const Toggle& toggle : toggles)
    {
        PostProcessSettings settings = all;
        settings.*(toggle.flag) = false;

        PostProcessTimings off{};
        CG_REQUIRE(
            stack.Execute(inputs, destination, settings, view, 1.0f / 60.0f, nullptr, &off).IsOk());

        const PassTiming& timing = off.passes[static_cast<i32>(toggle.pass)];
        CG_EXPECT_FALSE(timing.executed);
        CG_EXPECT_EQ(timing.nanoseconds, 0u);

        ImageDifferenceMetrics difference{};
        CG_REQUIRE(MeasureImageDifference(destination, reference, 1.0f / 255.0f, difference).IsOk());
        std::printf("[phase30] toggle %-18s off: mean %.6f, differing %.4f\n",
                    PostProcessPassName(toggle.pass),
                    static_cast<double>(difference.meanAbsoluteError),
                    static_cast<double>(difference.differingFraction));
        CG_EXPECT_TRUE(difference.differingFraction > 0.01f);
    }

    // Exposure is measured by its metered value rather than by an image
    // difference, because a manual exposure of 0 EV may land on the same
    // image the meter chose.
    PostProcessSettings withoutExposure = all;
    withoutExposure.enableExposure = false;
    PostProcessTimings exposureOff{};
    CG_REQUIRE(stack
                   .Execute(inputs, destination, withoutExposure, view, 1.0f / 60.0f, nullptr,
                            &exposureOff)
                   .IsOk());
    CG_EXPECT_FALSE(exposureOff.passes[static_cast<i32>(PostProcessPass::Exposure)].executed);
    CG_EXPECT_NEAR(exposureOff.exposureEv, 0.0f, 1.0e-6f);
    CG_EXPECT_TRUE(std::abs(timings.exposureEv) > 0.1f);
}

// ---------------------------------------------------------------------------
// Done-when, part two: no visible ghosting in the moving scene.
// ---------------------------------------------------------------------------

CG_TEST(TemporalAntiAliasingDoesNotGhostInTheMovingScene)
{
    HeapAllocator heap;

    // The shipped configuration is the default one: there is no second set of
    // settings that the acceptance result was obtained with.
    const TemporalAaSettings settings{};

    const GhostingRun run = RunScrollingScene(heap, settings, true, 24);
    CG_REQUIRE(run.ok);
    std::printf("[phase30] ghosting: changed=%u (%.4f) weighted=%.6f mean=%.6f max=%.6f "
                "ghosted=%u (%.6f)\n",
                run.metrics.changedPixels, static_cast<double>(run.metrics.changedFraction),
                static_cast<double>(run.metrics.weightedContamination),
                static_cast<double>(run.metrics.meanContamination),
                static_cast<double>(run.metrics.maxContamination), run.metrics.ghostedPixels,
                static_cast<double>(run.metrics.ghostedFraction));

    // The scene has to have moved, or the measurement is vacuous.
    CG_EXPECT_TRUE(run.metrics.changedFraction > 0.2f);

    // Ghost-free, as a number: fitted over the whole moving image, at most a
    // twentieth of the previous frame survives into this one.
    //
    // The threshold is absolute rather than a ratio against a reference run.
    // The run that would define such a ratio is a frame carrying no history at
    // all, and that frame fits zero by construction -- its error is sampling
    // noise, which points nowhere in particular -- so no multiple of it is a
    // threshold. What calibrates an absolute number instead is the control
    // below: the same scene with the two mechanisms that prevent trailing
    // switched off, which has to miss this threshold by a wide margin for it
    // to mean anything.
    CG_EXPECT_TRUE(run.metrics.weightedContamination < 0.05f);

    // The per-pixel ratios for the same run. They are looser because each one
    // divides by how far its own pixel moved, so a pixel that barely cleared
    // the change threshold reports a large ratio for a negligible error.
    CG_EXPECT_TRUE(run.metrics.meanContamination < 0.10f);
    CG_EXPECT_TRUE(run.metrics.ghostedFraction < 0.15f);

    // The floor: history weight zero, so the resolve is exactly the current
    // jittered frame and cannot ghost. It fits no part of the previous frame,
    // which is what makes the fitted metric the one to assert on; its
    // per-pixel mean is not zero, and that residue is the per-pixel metric's
    // own noise floor for this scene.
    TemporalAaSettings noHistory = settings;
    noHistory.historyWeight = 0.0f;
    const GhostingRun floorRun = RunScrollingScene(heap, noHistory, true, 24);
    CG_REQUIRE(floorRun.ok);
    std::printf("[phase30] ghosting floor (no accumulation): weighted=%.6f mean=%.6f "
                "ghosted=%.6f\n",
                static_cast<double>(floorRun.metrics.weightedContamination),
                static_cast<double>(floorRun.metrics.meanContamination),
                static_cast<double>(floorRun.metrics.ghostedFraction));
    CG_EXPECT_TRUE(floorRun.metrics.weightedContamination < 0.01f);

    // How the clip box width trades against the measurement, reported so the
    // default is chosen from data rather than taste.
    for (const f32 gamma : {0.75f, 1.0f, 1.5f, 2.0f, 3.0f})
    {
        TemporalAaSettings variant = settings;
        variant.varianceGamma = gamma;
        const GhostingRun variantRun = RunScrollingScene(heap, variant, true, 24);
        CG_REQUIRE(variantRun.ok);
        std::printf("[phase30] ghosting at variance gamma %.2f: weighted=%.6f mean=%.6f "
                    "ghosted=%.6f\n",
                    static_cast<double>(gamma),
                    static_cast<double>(variantRun.metrics.weightedContamination),
                    static_cast<double>(variantRun.metrics.meanContamination),
                    static_cast<double>(variantRun.metrics.ghostedFraction));
    }

    // And the same for the flicker relaxation, the knob that decides how much
    // of a clamped sub-pixel feature is held onto.
    //
    // Widening the box is the direction that trusts history more, so this is
    // the sweep that would show trailing creeping back in if it were going to.
    // It does the opposite: contamination falls monotonically as the box
    // widens. That is not an accident of the metric. A strict clamp does not
    // keep stale color out, it throws live color away -- the half-pixel lines
    // the current sample missed -- and the hole it leaves lies along the same
    // axis a trail would, because that is where the previous frame's lines
    // were. Holding onto the feature removes the artifact rather than adding
    // one.
    f32 previousContamination = 1.0f;
    for (const f32 relaxation : {0.0f, 1.0f, 2.0f, 4.0f})
    {
        TemporalAaSettings variant = settings;
        variant.flickerRelaxation = relaxation;
        const GhostingRun variantRun = RunScrollingScene(heap, variant, true, 24);
        CG_REQUIRE(variantRun.ok);
        std::printf("[phase30] ghosting at flicker relaxation %.2f: weighted=%.6f mean=%.6f "
                    "ghosted=%.6f\n",
                    static_cast<double>(relaxation),
                    static_cast<double>(variantRun.metrics.weightedContamination),
                    static_cast<double>(variantRun.metrics.meanContamination),
                    static_cast<double>(variantRun.metrics.ghostedFraction));
        CG_EXPECT_TRUE(variantRun.metrics.weightedContamination <= previousContamination);
        previousContamination = variantRun.metrics.weightedContamination;
        // Every setting that is offered as a shipping one clears the
        // threshold. Relaxation 0 is not one: it is the strict neighborhood
        // clamp, which the convergence test keeps as the control that loses
        // sub-pixel detail, and it sits right at this threshold too.
        if (relaxation >= 1.0f)
        {
            CG_EXPECT_TRUE(variantRun.metrics.weightedContamination < 0.05f);
        }
    }

    // Control: the same scene with history neither reprojected nor clamped.
    // This is the configuration that produces the artifact, and it has to fail
    // by a wide margin what the shipped configuration passes.
    TemporalAaSettings control = settings;
    control.useVelocity = false;
    control.clampHistory = false;
    const GhostingRun controlRun = RunScrollingScene(heap, control, false, 24);
    CG_REQUIRE(controlRun.ok);
    std::printf("[phase30] ghosting control (no reprojection, no clamp): weighted=%.6f mean=%.6f "
                "ghosted=%.6f\n",
                static_cast<double>(controlRun.metrics.weightedContamination),
                static_cast<double>(controlRun.metrics.meanContamination),
                static_cast<double>(controlRun.metrics.ghostedFraction));
    CG_EXPECT_TRUE(controlRun.metrics.weightedContamination > 0.40f);
    CG_EXPECT_TRUE(controlRun.metrics.weightedContamination >
                   run.metrics.weightedContamination * 10.0f);
    CG_EXPECT_TRUE(controlRun.metrics.meanContamination > run.metrics.meanContamination * 2.0f);
    CG_EXPECT_TRUE(controlRun.metrics.ghostedFraction > run.metrics.ghostedFraction * 2.0f);

    // Reprojection without the clamp is measured too, and reported rather than
    // asserted on. With a velocity buffer this exact it is the better of the
    // two on this metric; the clamp earns its place on the disocclusions the
    // control above shows and on the camera cut the unit suite covers, not
    // here.
    TemporalAaSettings unclamped = settings;
    unclamped.clampHistory = false;
    const GhostingRun unclampedRun = RunScrollingScene(heap, unclamped, true, 24);
    CG_REQUIRE(unclampedRun.ok);
    std::printf("[phase30] ghosting variant (reprojection, no clamp): weighted=%.6f mean=%.6f "
                "ghosted=%.6f\n",
                static_cast<double>(unclampedRun.metrics.weightedContamination),
                static_cast<double>(unclampedRun.metrics.meanContamination),
                static_cast<double>(unclampedRun.metrics.ghostedFraction));
    CG_EXPECT_TRUE(unclampedRun.metrics.meanContamination < controlRun.metrics.meanContamination);
}

CG_TEST(TemporalAntiAliasingIsStableOnAStillCamera)
{
    HeapAllocator heap;

    Framebuffer jittered;
    CG_REQUIRE(MakeImage(heap, jittered, kWidth, kHeight));

    Framebuffer resolved[6];
    Framebuffer raw[6];
    for (i32 index = 0; index < 6; ++index)
    {
        CG_REQUIRE(MakeImage(heap, resolved[index], kWidth, kHeight));
        CG_REQUIRE(MakeImage(heap, raw[index], kWidth, kHeight));
    }

    auto created = TemporalAaResolver::Create(kWidth, kHeight, heap);
    CG_REQUIRE(created.IsOk());
    TemporalAaResolver resolver = std::move(created).Value();

    TemporalAaSettings settings{};
    constexpr i32 kWarmup = 24;
    for (i32 frame = 0; frame < kWarmup + 6; ++frame)
    {
        RenderJittered(jittered, 0.0f, TemporalJitterOffset(static_cast<u32>(frame)));
        const i32 slot = frame - kWarmup;
        if (slot >= 0)
        {
            // The unresolved frame is the control: one jittered sample per
            // pixel flickers precisely because the jitter moves.
            for (i32 y = 0; y < kHeight; ++y)
            {
                for (i32 x = 0; x < kWidth; ++x)
                {
                    raw[slot].SetPixel(x, y, jittered.GetPixel(x, y));
                }
            }
        }

        Framebuffer scratch;
        CG_REQUIRE(MakeImage(heap, scratch, kWidth, kHeight));
        CG_REQUIRE(resolver.Resolve(jittered, nullptr, scratch, settings, nullptr, nullptr).IsOk());
        if (slot >= 0)
        {
            for (i32 y = 0; y < kHeight; ++y)
            {
                for (i32 x = 0; x < kWidth; ++x)
                {
                    resolved[slot].SetPixel(x, y, scratch.GetPixel(x, y));
                }
            }
        }
    }

    const Framebuffer* resolvedFrames[6] = {&resolved[0], &resolved[1], &resolved[2],
                                            &resolved[3], &resolved[4], &resolved[5]};
    const Framebuffer* rawFrames[6] = {&raw[0], &raw[1], &raw[2], &raw[3], &raw[4], &raw[5]};

    TemporalStabilityMetrics stable{};
    TemporalStabilityMetrics control{};
    CG_REQUIRE(MeasureTemporalStability(Span<const Framebuffer* const>{resolvedFrames, 6}, stable)
                   .IsOk());
    CG_REQUIRE(
        MeasureTemporalStability(Span<const Framebuffer* const>{rawFrames, 6}, control).IsOk());

    std::printf("[phase30] stability: resolved mean=%.6f max=%.6f drift=%.6f; "
                "unresolved mean=%.6f max=%.6f\n",
                static_cast<double>(stable.meanFrameDelta),
                static_cast<double>(stable.maxFrameDelta), static_cast<double>(stable.totalDrift),
                static_cast<double>(control.meanFrameDelta),
                static_cast<double>(control.maxFrameDelta));

    // A converged still camera must be nearly frozen, and far steadier than
    // the jittered frames it was accumulated from.
    CG_EXPECT_TRUE(stable.meanFrameDelta < 0.01f);
    CG_EXPECT_TRUE(stable.meanFrameDelta < control.meanFrameDelta * 0.25f);
}

CG_TEST(TemporalAntiAliasingRemovesTheStaircaseItWasGivenToRemove)
{
    HeapAllocator heap;

    Framebuffer jittered;
    Framebuffer resolved;
    Framebuffer reference;
    CG_REQUIRE(MakeImage(heap, jittered, kWidth, kHeight));
    CG_REQUIRE(MakeImage(heap, resolved, kWidth, kHeight));
    CG_REQUIRE(MakeImage(heap, reference, kWidth, kHeight));

    auto created = TemporalAaResolver::Create(kWidth, kHeight, heap);
    CG_REQUIRE(created.IsOk());
    TemporalAaResolver resolver = std::move(created).Value();

    TemporalAaSettings settings{};
    for (i32 frame = 0; frame < 32; ++frame)
    {
        RenderJittered(jittered, 0.0f, TemporalJitterOffset(static_cast<u32>(frame)));
        CG_REQUIRE(resolver.Resolve(jittered, nullptr, resolved, settings, nullptr, nullptr).IsOk());
    }
    RenderReference(reference, 0.0f);

    EdgeAliasingMetrics aliased{};
    EdgeAliasingMetrics converged{};
    EdgeAliasingMetrics truth{};
    CG_REQUIRE(MeasureEdgeAliasing(jittered, 0.25f, aliased).IsOk());
    CG_REQUIRE(MeasureEdgeAliasing(resolved, 0.25f, converged).IsOk());
    CG_REQUIRE(MeasureEdgeAliasing(reference, 0.25f, truth).IsOk());

    ImageDifferenceMetrics againstTruth{};
    ImageDifferenceMetrics rawAgainstTruth{};
    CG_REQUIRE(MeasureImageDifference(resolved, reference, 1.0f / 255.0f, againstTruth).IsOk());
    CG_REQUIRE(MeasureImageDifference(jittered, reference, 1.0f / 255.0f, rawAgainstTruth).IsOk());

    std::printf("[phase30] edges: one sample %.6f, temporal %.6f, reference %.6f\n",
                static_cast<double>(aliased.hardStepFraction),
                static_cast<double>(converged.hardStepFraction),
                static_cast<double>(truth.hardStepFraction));
    std::printf("[phase30] against the 8x8 reference: one sample %.6f, temporal %.6f\n",
                static_cast<double>(rawAgainstTruth.meanAbsoluteError),
                static_cast<double>(againstTruth.meanAbsoluteError));

    // Accumulating the jitter has to converge on the integrated image, both in
    // edge statistics and pixel for pixel.
    CG_EXPECT_TRUE(converged.hardStepFraction < aliased.hardStepFraction);
    CG_EXPECT_TRUE(againstTruth.meanAbsoluteError < rawAgainstTruth.meanAbsoluteError * 0.5f);

    // Converging is not the same as smoothing, and the edge count alone cannot
    // tell them apart: an image blurred until nothing is left also has no hard
    // steps. The resolve has to land on the reference's own edge count, not
    // below it, which is the claim that the detail it kept is the detail that
    // is there.
    CG_EXPECT_TRUE(converged.hardStepFraction > truth.hardStepFraction * 0.75f);

    // The control is the same accumulation with the flicker relaxation off, so
    // that the clip box is the strict neighborhood of a one-sample frame. It
    // is the configuration that loses the thin lines: it converges to a
    // measurably worse image and to measurably fewer edges than the reference
    // has, which is what a sub-pixel feature clamped away looks like.
    TemporalAaSettings strict{};
    strict.flickerRelaxation = 0.0f;
    auto strictCreated = TemporalAaResolver::Create(kWidth, kHeight, heap);
    CG_REQUIRE(strictCreated.IsOk());
    TemporalAaResolver strictResolver = std::move(strictCreated).Value();
    Framebuffer strictResolved;
    CG_REQUIRE(MakeImage(heap, strictResolved, kWidth, kHeight));
    for (i32 frame = 0; frame < 32; ++frame)
    {
        RenderJittered(jittered, 0.0f, TemporalJitterOffset(static_cast<u32>(frame)));
        CG_REQUIRE(
            strictResolver.Resolve(jittered, nullptr, strictResolved, strict, nullptr, nullptr)
                .IsOk());
    }
    EdgeAliasingMetrics strictEdges{};
    ImageDifferenceMetrics strictAgainstTruth{};
    CG_REQUIRE(MeasureEdgeAliasing(strictResolved, 0.25f, strictEdges).IsOk());
    CG_REQUIRE(
        MeasureImageDifference(strictResolved, reference, 1.0f / 255.0f, strictAgainstTruth).IsOk());
    std::printf("[phase30] strict clamp control: error %.6f, edges %.6f\n",
                static_cast<double>(strictAgainstTruth.meanAbsoluteError),
                static_cast<double>(strictEdges.hardStepFraction));
    CG_EXPECT_TRUE(strictAgainstTruth.meanAbsoluteError > rawAgainstTruth.meanAbsoluteError * 0.5f);
    CG_EXPECT_TRUE(strictEdges.hardStepFraction < truth.hardStepFraction * 0.75f);
}
