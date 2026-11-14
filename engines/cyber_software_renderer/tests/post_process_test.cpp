#include <cyber/jobs/job_system.h>
#include <cyber/math/mat4.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/software/post_process.h>
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

namespace
{

constexpr i32 kWidth = 48;
constexpr i32 kHeight = 32;

[[nodiscard]] bool MakeImage(HeapAllocator& heap, Framebuffer& out, i32 width, i32 height,
                             PixelFormat format = PixelFormat::RGBA32F)
{
    FramebufferDesc desc{};
    desc.width = width;
    desc.height = height;
    desc.format = format;
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

// Reverse-Z depth for a view distance, the inverse of LinearizeReverseZ.
[[nodiscard]] f32 DepthForDistance(f32 distance, f32 nearPlane, f32 farPlane)
{
    return (nearPlane * farPlane / distance - nearPlane) / (farPlane - nearPlane);
}

[[nodiscard]] PostProcessView MakeView(f32 nearPlane = 0.1f, f32 farPlane = 100.0f)
{
    PostProcessView view{};
    view.projection = Mat4::PerspectiveReverseZ(
        1.0471975512f, static_cast<f32>(kWidth) / static_cast<f32>(kHeight), nearPlane, farPlane);
    view.view = Mat4::Identity();
    view.nearPlane = nearPlane;
    view.farPlane = farPlane;
    return view;
}

[[nodiscard]] bool ImagesEqual(const Framebuffer& lhs, const Framebuffer& rhs)
{
    if (lhs.Width() != rhs.Width() || lhs.Height() != rhs.Height())
    {
        return false;
    }
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

[[nodiscard]] f32 MeanLuminance(const Framebuffer& image)
{
    double total = 0.0;
    for (i32 y = 0; y < image.Height(); ++y)
    {
        for (i32 x = 0; x < image.Width(); ++x)
        {
            total += static_cast<double>(Luminance(image.GetPixel(x, y)));
        }
    }
    const double count = static_cast<double>(image.Width()) * static_cast<double>(image.Height());
    return static_cast<f32>(total / count);
}

}  // namespace

CG_TEST(LuminanceMatchesRec709Weights)
{
    CG_EXPECT_NEAR(Luminance(Color{1.0f, 0.0f, 0.0f, 1.0f}), 0.2126f, 1.0e-6f);
    CG_EXPECT_NEAR(Luminance(Color{0.0f, 1.0f, 0.0f, 1.0f}), 0.7152f, 1.0e-6f);
    CG_EXPECT_NEAR(Luminance(Color{0.0f, 0.0f, 1.0f, 1.0f}), 0.0722f, 1.0e-6f);
    CG_EXPECT_NEAR(Luminance(Color{1.0f, 1.0f, 1.0f, 1.0f}), 1.0f, 1.0e-6f);
}

CG_TEST(ReverseZLinearizesToTheDepthRange)
{
    constexpr f32 kNear = 0.25f;
    constexpr f32 kFar = 120.0f;
    CG_EXPECT_NEAR(LinearizeReverseZ(1.0f, kNear, kFar), kNear, 1.0e-4f);
    CG_EXPECT_NEAR(LinearizeReverseZ(0.0f, kNear, kFar), kFar, 1.0e-3f);

    // A round trip through the projection convention must return the distance.
    for (const f32 distance : {0.5f, 2.0f, 10.0f, 80.0f})
    {
        const f32 depth = DepthForDistance(distance, kNear, kFar);
        CG_EXPECT_NEAR(LinearizeReverseZ(depth, kNear, kFar), distance, distance * 1.0e-3f);
    }

    // A cleared or poisoned sample reads as the far plane rather than a NaN.
    CG_EXPECT_NEAR(LinearizeReverseZ(std::nanf(""), kNear, kFar), kFar, 1.0e-3f);
}

CG_TEST(ViewGeometryRecoversTheFieldOfView)
{
    constexpr f32 kFovY = 1.0471975512f;
    constexpr f32 kAspect = 1.5f;
    const PostProcessView view = MakeView();
    const ViewGeometry geometry = MakeViewGeometry(view);
    CG_EXPECT_NEAR(geometry.tanHalfFovY, std::tan(kFovY * 0.5f), 1.0e-4f);
    CG_EXPECT_NEAR(geometry.tanHalfFovX, std::tan(kFovY * 0.5f) * kAspect, 1.0e-3f);

    // An unset projection must not produce NaNs downstream.
    PostProcessView degenerate{};
    degenerate.projection = Mat4::Zero();
    const ViewGeometry fallback = MakeViewGeometry(degenerate);
    CG_EXPECT_TRUE(std::isfinite(fallback.tanHalfFovX));
    CG_EXPECT_TRUE(std::isfinite(fallback.tanHalfFovY));
}

CG_TEST(ReconstructedPositionProjectsBackToItsPixel)
{
    const ViewGeometry geometry = MakeViewGeometry(MakeView());
    const i32 pixels[4][2] = {{0, 0}, {kWidth - 1, 0}, {kWidth / 2, kHeight / 2}, {7, kHeight - 1}};
    for (const auto& pixel : pixels)
    {
        constexpr f32 kDistance = 4.0f;
        const auto position =
            ReconstructViewPosition(pixel[0], pixel[1], kWidth, kHeight, kDistance, geometry);
        CG_EXPECT_NEAR(position.z, kDistance, 1.0e-4f);

        const f32 ndcX = position.x / (position.z * geometry.tanHalfFovX);
        const f32 ndcY = position.y / (position.z * geometry.tanHalfFovY);
        const f32 screenX = (ndcX * 0.5f + 0.5f) * static_cast<f32>(kWidth) - 0.5f;
        const f32 screenY = (0.5f - ndcY * 0.5f) * static_cast<f32>(kHeight) - 0.5f;
        CG_EXPECT_NEAR(screenX, static_cast<f32>(pixel[0]), 1.0e-3f);
        CG_EXPECT_NEAR(screenY, static_cast<f32>(pixel[1]), 1.0e-3f);
    }
}

CG_TEST(HistogramCountsEveryPixelExactlyOnce)
{
    HeapAllocator heap;
    Framebuffer image;
    CG_REQUIRE(MakeImage(heap, image, 16, 8));
    image.Clear(Color{0.25f, 0.25f, 0.25f, 1.0f});
    image.SetPixel(0, 0, Color{0.0f, 0.0f, 0.0f, 1.0f});
    image.SetPixel(1, 0, Color{std::nanf(""), 0.0f, 0.0f, 1.0f});
    image.SetPixel(2, 0, Color{1.0e6f, 1.0e6f, 1.0e6f, 1.0f});

    LuminanceHistogram histogram{};
    CG_REQUIRE(BuildLuminanceHistogram(image, -10.0f, 10.0f, histogram).IsOk());
    CG_EXPECT_EQ(histogram.totalSamples, 128u);

    u32 counted = 0;
    for (const u32 bin : histogram.bins)
    {
        counted += bin;
    }
    CG_EXPECT_EQ(counted, histogram.totalSamples);

    // Black and the non-finite pixel land in the first bin; the overbright one
    // saturates into the last.
    CG_EXPECT_TRUE(histogram.bins[0] >= 2u);
    CG_EXPECT_TRUE(histogram.bins[kLuminanceHistogramBins - 1] >= 1u);

    CG_EXPECT_FALSE(BuildLuminanceHistogram(image, 1.0f, 1.0f, histogram).IsOk());
}

CG_TEST(MeteredExposureCentersOnMiddleGrey)
{
    HeapAllocator heap;
    Framebuffer image;
    CG_REQUIRE(MakeImage(heap, image, 16, 16));

    // A uniform frame four stops above middle grey must meter four stops down.
    image.Clear(Color{kMiddleGrey * 16.0f, kMiddleGrey * 16.0f, kMiddleGrey * 16.0f, 1.0f});
    LuminanceHistogram histogram{};
    CG_REQUIRE(BuildLuminanceHistogram(image, -10.0f, 10.0f, histogram).IsOk());
    const f32 average = HistogramAverageLuminance(histogram, 0.0f, 1.0f);
    CG_EXPECT_NEAR(average, kMiddleGrey * 16.0f, kMiddleGrey * 16.0f * 0.2f);
    CG_EXPECT_NEAR(ExposureEvForAverageLuminance(average), -4.0f, 0.3f);

    // An empty window is a no-op rather than a jump to black.
    CG_EXPECT_NEAR(HistogramAverageLuminance(histogram, 0.9f, 0.1f), kMiddleGrey, 1.0e-6f);
    CG_EXPECT_NEAR(ExposureEvForAverageLuminance(0.0f), 0.0f, 1.0e-6f);
}

CG_TEST(BrightWindowDoesNotDragTheWholeFrame)
{
    HeapAllocator heap;
    Framebuffer image;
    CG_REQUIRE(MakeImage(heap, image, 32, 32));
    image.Clear(Color{0.18f, 0.18f, 0.18f, 1.0f});
    // A small, very bright window: 4% of the frame at 1000x middle grey.
    image.ClearRect(0, 0, 6, 7, Color{180.0f, 180.0f, 180.0f, 1.0f});

    LuminanceHistogram histogram{};
    CG_REQUIRE(BuildLuminanceHistogram(image, -10.0f, 10.0f, histogram).IsOk());
    const f32 windowed = HistogramAverageLuminance(histogram, 0.5f, 0.95f);
    const f32 unwindowed = HistogramAverageLuminance(histogram, 0.0f, 1.0f);
    CG_EXPECT_TRUE(windowed < unwindowed);
    CG_EXPECT_NEAR(windowed, 0.18f, 0.05f);
}

CG_TEST(ExposureAdaptationRampsAndSnapsOnTheFirstFrame)
{
    ExposureSettings settings{};
    settings.adaptUpEvPerSecond = 3.0f;
    settings.adaptDownEvPerSecond = 1.0f;

    ExposureState state{};
    CG_EXPECT_NEAR(AdaptExposure(state, 4.0f, 1.0f / 60.0f, settings), 4.0f, 1.0e-6f);
    CG_EXPECT_TRUE(state.adapted);

    // Brightening is faster than darkening, by configuration.
    state = ExposureState{};
    state.adapted = true;
    state.currentEv = 0.0f;
    const f32 up = AdaptExposure(state, 10.0f, 0.5f, settings);
    CG_EXPECT_NEAR(up, 1.5f, 1.0e-5f);

    state.currentEv = 0.0f;
    const f32 down = AdaptExposure(state, -10.0f, 0.5f, settings);
    CG_EXPECT_NEAR(down, -0.5f, 1.0e-5f);

    // The ramp lands exactly on the target instead of overshooting.
    state.currentEv = 0.0f;
    CG_EXPECT_NEAR(AdaptExposure(state, 0.25f, 1.0f, settings), 0.25f, 1.0e-6f);

    // A stalled frame snaps rather than freezing mid-ramp.
    state.currentEv = 0.0f;
    CG_EXPECT_NEAR(AdaptExposure(state, 2.0f, 0.0f, settings), 2.0f, 1.0e-6f);
}

CG_TEST(ManualExposureIgnoresTheScene)
{
    HeapAllocator heap;
    Framebuffer image;
    CG_REQUIRE(MakeImage(heap, image, 8, 8));
    image.Clear(Color{50.0f, 50.0f, 50.0f, 1.0f});

    ExposureSettings settings{};
    settings.mode = ExposureMode::Manual;
    settings.manualEv = -2.5f;
    ExposureState state{};
    auto computed = ComputeExposureEv(image, settings, state, 1.0f / 60.0f);
    CG_REQUIRE(computed.IsOk());
    CG_EXPECT_NEAR(computed.Value(), -2.5f, 1.0e-6f);
}

CG_TEST(MeteredExposureRespectsItsBoundsAndCompensation)
{
    HeapAllocator heap;
    Framebuffer image;
    CG_REQUIRE(MakeImage(heap, image, 16, 16));
    image.Clear(Color{0.0001f, 0.0001f, 0.0001f, 1.0f});

    ExposureSettings settings{};
    settings.mode = ExposureMode::Automatic;
    settings.minEv = -2.0f;
    settings.maxEv = 3.0f;
    settings.compensationEv = 0.5f;
    ExposureState state{};
    auto computed = ComputeExposureEv(image, settings, state, 1.0f / 60.0f);
    CG_REQUIRE(computed.IsOk());
    // A nearly black frame wants far more than +3 EV; the clamp holds it, and
    // compensation is applied on top of the clamped value.
    CG_EXPECT_NEAR(computed.Value(), 3.5f, 1.0e-5f);
}

CG_TEST(ToneMapOperatorsRollOffWithoutClipping)
{
    ToneMapSettings aces{};
    aces.op = ToneMapOperator::AcesFitted;
    ToneMapSettings reinhard{};
    reinhard.op = ToneMapOperator::Reinhard;
    ToneMapSettings filmic{};
    filmic.op = ToneMapOperator::Filmic;
    ToneMapSettings clamp{};
    clamp.op = ToneMapOperator::Clamp;

    // The control clips: two very different highlights become the same white.
    const Color clipLow = ApplyToneMap(Color{1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, clamp);
    const Color clipHigh = ApplyToneMap(Color{2.5f, 2.5f, 2.5f, 1.0f}, 0.0f, clamp);
    CG_EXPECT_NEAR(clipLow.r, 1.0f, 1.0e-6f);
    CG_EXPECT_NEAR(clipHigh.r, 1.0f, 1.0e-6f);

    for (const ToneMapSettings& settings : {aces, reinhard, filmic})
    {
        f32 previous = -1.0f;
        for (const f32 value : {0.0f, 0.05f, 0.18f, 0.5f, 1.0f, 2.0f, 8.0f, 64.0f})
        {
            const Color mapped = ApplyToneMap(Color{value, value, value, 1.0f}, 0.0f, settings);
            CG_EXPECT_TRUE(mapped.r >= 0.0f && mapped.r <= 1.0f);
            // Monotonic: a brighter scene value is never darker on screen.
            CG_EXPECT_TRUE(mapped.r >= previous);
            previous = mapped.r;
        }
        // Those two highlights stay distinguishable instead of both reading as
        // white, which is the whole point of a roll-off.
        const Color low = ApplyToneMap(Color{1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, settings);
        const Color high = ApplyToneMap(Color{2.5f, 2.5f, 2.5f, 1.0f}, 0.0f, settings);
        CG_EXPECT_TRUE(high.r - low.r > 0.01f);
    }

    // Exposure is a scale in stops: +1 EV doubles the scene before the curve.
    const Color base = ApplyToneMap(Color{0.1f, 0.1f, 0.1f, 1.0f}, 1.0f, clamp);
    CG_EXPECT_NEAR(base.r, 0.2f, 1.0e-5f);

    // Alpha passes through and a poisoned channel resolves to black.
    const Color poisoned =
        ApplyToneMap(Color{std::nanf(""), 0.0f, 0.0f, 0.25f}, 0.0f, aces);
    CG_EXPECT_NEAR(poisoned.a, 0.25f, 1.0e-6f);
    CG_EXPECT_TRUE(std::isfinite(poisoned.r));
}

CG_TEST(IdentityGradeIsANoOp)
{
    const ColorGradeSettings identity{};
    for (const f32 value : {0.0f, 0.18f, 0.5f, 1.0f})
    {
        const Color graded = ApplyColorGrade(Color{value, value * 0.5f, value * 0.25f, 1.0f},
                                             identity);
        CG_EXPECT_NEAR(graded.r, value, 1.0e-5f);
        CG_EXPECT_NEAR(graded.g, value * 0.5f, 1.0e-5f);
        CG_EXPECT_NEAR(graded.b, value * 0.25f, 1.0e-5f);
    }
}

CG_TEST(GradeControlsMoveTheExpectedDirection)
{
    const Color source{0.4f, 0.4f, 0.4f, 1.0f};

    ColorGradeSettings saturated{};
    saturated.saturation = 0.0f;
    const Color grey = ApplyColorGrade(Color{0.8f, 0.2f, 0.1f, 1.0f}, saturated);
    CG_EXPECT_NEAR(grey.r, grey.g, 1.0e-5f);
    CG_EXPECT_NEAR(grey.g, grey.b, 1.0e-5f);

    ColorGradeSettings contrast{};
    contrast.contrast = 2.0f;
    // Above middle grey, more contrast is brighter; below it, darker.
    CG_EXPECT_TRUE(ApplyColorGrade(source, contrast).r > source.r);
    CG_EXPECT_TRUE(ApplyColorGrade(Color{0.1f, 0.1f, 0.1f, 1.0f}, contrast).r < 0.1f);

    ColorGradeSettings warm{};
    warm.temperature = 1.0f;
    const Color warmed = ApplyColorGrade(source, warm);
    CG_EXPECT_TRUE(warmed.r > warmed.b);

    ColorGradeSettings gain{};
    gain.gain = Color{0.5f, 1.0f, 1.0f, 1.0f};
    CG_EXPECT_TRUE(ApplyColorGrade(source, gain).r < source.r);
}

CG_TEST(IdentityLutReproducesItsInput)
{
    HeapAllocator heap;
    auto created = ColorGradeLut::Identity(kDefaultColorGradeLutSize, heap);
    CG_REQUIRE(created.IsOk());
    const ColorGradeLut lut = std::move(created).Value();
    CG_EXPECT_TRUE(lut.IsValid());
    CG_EXPECT_EQ(lut.Size(), kDefaultColorGradeLutSize);
    CG_EXPECT_EQ(lut.Entries().Size(),
                 static_cast<cyber::core::usize>(kDefaultColorGradeLutSize) *
                     static_cast<cyber::core::usize>(kDefaultColorGradeLutSize) *
                     static_cast<cyber::core::usize>(kDefaultColorGradeLutSize));

    for (i32 step = 0; step <= 20; ++step)
    {
        const f32 value = static_cast<f32>(step) / 20.0f;
        const Color sampled = lut.Sample(Color{value, 1.0f - value, value * 0.5f, 0.5f});
        CG_EXPECT_NEAR(sampled.r, value, 1.0e-3f);
        CG_EXPECT_NEAR(sampled.g, 1.0f - value, 1.0e-3f);
        CG_EXPECT_NEAR(sampled.b, value * 0.5f, 1.0e-3f);
        CG_EXPECT_NEAR(sampled.a, 0.5f, 1.0e-6f);
    }
}

CG_TEST(BakedLutMatchesTheGradeItWasBakedFrom)
{
    HeapAllocator heap;
    ColorGradeSettings settings{};
    settings.contrast = 1.3f;
    settings.saturation = 0.7f;
    settings.temperature = 0.3f;
    settings.gain = Color{1.1f, 1.0f, 0.9f, 1.0f};

    auto created = ColorGradeLut::Bake(settings, 48, heap);
    CG_REQUIRE(created.IsOk());
    const ColorGradeLut lut = std::move(created).Value();

    f32 worst = 0.0f;
    for (i32 step = 0; step <= 16; ++step)
    {
        const f32 value = static_cast<f32>(step) / 16.0f;
        const Color source{value, 1.0f - value, 0.5f, 1.0f};
        const Color direct = ApplyColorGrade(source, settings);
        const Color sampled = lut.Sample(source);
        worst = std::max({worst, std::abs(direct.r - sampled.r), std::abs(direct.g - sampled.g),
                          std::abs(direct.b - sampled.b)});
    }
    // Trilinear reconstruction of a 48-cube: the error is bounded by the
    // lattice spacing, not by the grade.
    CG_EXPECT_TRUE(worst < 0.01f);

    CG_EXPECT_FALSE(ColorGradeLut::Bake(settings, 1, heap).IsOk());
    CG_EXPECT_FALSE(ColorGradeLut::Bake(settings, 128, heap).IsOk());
}

CG_TEST(StackRejectsMismatchedBuffers)
{
    HeapAllocator heap;
    auto created = PostProcessStack::Create(kWidth, kHeight, heap);
    CG_REQUIRE(created.IsOk());
    PostProcessStack stack = std::move(created).Value();
    CG_EXPECT_TRUE(stack.IsValid());
    CG_EXPECT_EQ(stack.Width(), kWidth);
    CG_EXPECT_EQ(stack.Height(), kHeight);

    Framebuffer color;
    CG_REQUIRE(MakeImage(heap, color, kWidth, kHeight));
    Framebuffer wrongSize;
    CG_REQUIRE(MakeImage(heap, wrongSize, kWidth / 2, kHeight));

    PostProcessInputs inputs{};
    inputs.color = &color;
    PostProcessSettings settings{};
    const PostProcessView view = MakeView();

    CG_EXPECT_FALSE(
        stack.Execute(inputs, wrongSize, settings, view, 1.0f / 60.0f, nullptr, nullptr).IsOk());

    inputs.color = &wrongSize;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, destination, kWidth, kHeight));
    CG_EXPECT_FALSE(
        stack.Execute(inputs, destination, settings, view, 1.0f / 60.0f, nullptr, nullptr).IsOk());

    inputs.color = nullptr;
    CG_EXPECT_FALSE(
        stack.Execute(inputs, destination, settings, view, 1.0f / 60.0f, nullptr, nullptr).IsOk());

    CG_EXPECT_FALSE(PostProcessStack::Create(0, kHeight, heap).IsOk());
}

CG_TEST(DisabledPassesAreReportedAndCostNothing)
{
    HeapAllocator heap;
    auto created = PostProcessStack::Create(kWidth, kHeight, heap);
    CG_REQUIRE(created.IsOk());
    PostProcessStack stack = std::move(created).Value();

    Framebuffer color;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, color, kWidth, kHeight));
    CG_REQUIRE(MakeImage(heap, destination, kWidth, kHeight));
    color.Clear(Color{0.18f, 0.18f, 0.18f, 1.0f});

    PostProcessInputs inputs{};
    inputs.color = &color;

    PostProcessSettings settings{};
    settings.enableAmbientOcclusion = false;
    settings.enableBloom = false;
    settings.enableExposure = false;
    settings.enableToneMap = false;
    settings.enableColorGrade = false;

    PostProcessTimings timings{};
    CG_REQUIRE(stack
                   .Execute(inputs, destination, settings, MakeView(), 1.0f / 60.0f, nullptr,
                            &timings)
                   .IsOk());

    for (const PassTiming& timing : timings.passes)
    {
        CG_EXPECT_FALSE(timing.executed);
        CG_EXPECT_EQ(timing.nanoseconds, 0u);
        CG_EXPECT_TRUE(timing.name != nullptr);
    }

    // With every pass off the chain is a copy, which is what makes a toggle
    // meaningful rather than approximate.
    CG_EXPECT_TRUE(ImagesEqual(color, destination));
}

CG_TEST(PassesWithoutTheirInputsAreSkippedNotFailed)
{
    HeapAllocator heap;
    auto created = PostProcessStack::Create(kWidth, kHeight, heap);
    CG_REQUIRE(created.IsOk());
    PostProcessStack stack = std::move(created).Value();

    Framebuffer color;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, color, kWidth, kHeight));
    CG_REQUIRE(MakeImage(heap, destination, kWidth, kHeight));
    color.Clear(Color{0.5f, 0.5f, 0.5f, 1.0f});

    PostProcessInputs inputs{};
    inputs.color = &color;

    // Ambient occlusion needs depth and normals; motion blur needs velocity.
    // None are supplied, and the frame still resolves.
    PostProcessSettings settings{};
    settings.enableAmbientOcclusion = true;
    settings.enableMotionBlur = true;
    settings.enableDepthOfField = true;
    settings.enableBloom = false;

    PostProcessTimings timings{};
    CG_REQUIRE(stack
                   .Execute(inputs, destination, settings, MakeView(), 1.0f / 60.0f, nullptr,
                            &timings)
                   .IsOk());
    CG_EXPECT_FALSE(timings.passes[static_cast<i32>(PostProcessPass::AmbientOcclusion)].executed);
    CG_EXPECT_FALSE(timings.passes[static_cast<i32>(PostProcessPass::MotionBlur)].executed);
    CG_EXPECT_FALSE(timings.passes[static_cast<i32>(PostProcessPass::DepthOfField)].executed);
    CG_EXPECT_TRUE(timings.passes[static_cast<i32>(PostProcessPass::ToneMap)].executed);
}

CG_TEST(BloomSpreadsEnergyAroundAHighlight)
{
    HeapAllocator heap;
    auto created = PostProcessStack::Create(64, 64, heap);
    CG_REQUIRE(created.IsOk());
    PostProcessStack stack = std::move(created).Value();

    Framebuffer color;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, color, 64, 64));
    CG_REQUIRE(MakeImage(heap, destination, 64, 64));
    color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    color.ClearRect(28, 28, 8, 8, Color{60.0f, 60.0f, 60.0f, 1.0f});

    PostProcessInputs inputs{};
    inputs.color = &color;

    PostProcessSettings settings{};
    settings.enableAmbientOcclusion = false;
    settings.enableBloom = true;
    settings.enableExposure = false;
    settings.enableToneMap = false;
    settings.bloom.intensity = 0.5f;
    settings.bloom.threshold = 1.0f;

    PostProcessTimings timings{};
    CG_REQUIRE(stack
                   .Execute(inputs, destination, settings, MakeView(), 1.0f / 60.0f, nullptr,
                            &timings)
                   .IsOk());
    CG_EXPECT_TRUE(timings.passes[static_cast<i32>(PostProcessPass::Bloom)].executed);

    // Far from the highlight the frame was black; bloom must have put light
    // there, and more of it nearer the source.
    const f32 nearGlow = Luminance(destination.GetPixel(20, 32));
    const f32 farGlow = Luminance(destination.GetPixel(4, 32));
    CG_EXPECT_TRUE(nearGlow > 1.0e-4f);
    CG_EXPECT_TRUE(nearGlow > farGlow);

    // With bloom off the same black pixel stays black.
    settings.enableBloom = false;
    CG_REQUIRE(stack
                   .Execute(inputs, destination, settings, MakeView(), 1.0f / 60.0f, nullptr,
                            &timings)
                   .IsOk());
    CG_EXPECT_NEAR(Luminance(destination.GetPixel(20, 32)), 0.0f, 1.0e-6f);
}

CG_TEST(AmbientOcclusionDarkensACreaseAndLeavesAFlatWallAlone)
{
    constexpr i32 kSize = 64;
    constexpr f32 kNear = 0.1f;
    constexpr f32 kFar = 50.0f;

    HeapAllocator heap;
    auto created = PostProcessStack::Create(kSize, kSize, heap);
    CG_REQUIRE(created.IsOk());
    PostProcessStack stack = std::move(created).Value();

    Framebuffer color;
    Framebuffer normals;
    Framebuffer destination;
    DepthBuffer depth;
    CG_REQUIRE(MakeImage(heap, color, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, normals, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, destination, kSize, kSize));
    CG_REQUIRE(MakeDepth(heap, depth, kSize, kSize));

    color.Clear(Color{0.5f, 0.5f, 0.5f, 1.0f});
    // A flat wall at 5 units, with a block standing 1.5 units closer over the
    // left half. The step between them is the crease.
    for (i32 y = 0; y < kSize; ++y)
    {
        for (i32 x = 0; x < kSize; ++x)
        {
            const f32 distance = x < kSize / 2 ? 3.5f : 5.0f;
            depth.TestAndWrite(x, y, DepthForDistance(distance, kNear, kFar));
            normals.SetPixel(x, y, Color{0.5f, 0.5f, 0.0f, 1.0f});
        }
    }

    PostProcessInputs inputs{};
    inputs.color = &color;
    inputs.depth = &depth;
    inputs.normals = &normals;

    PostProcessSettings settings{};
    settings.enableAmbientOcclusion = true;
    settings.enableBloom = false;
    settings.enableExposure = false;
    settings.enableToneMap = false;
    settings.ambientOcclusion.radius = 1.0f;
    settings.ambientOcclusion.intensity = 1.0f;

    PostProcessView view{};
    view.projection = Mat4::PerspectiveReverseZ(1.0471975512f, 1.0f, kNear, kFar);
    view.nearPlane = kNear;
    view.farPlane = kFar;

    PostProcessTimings timings{};
    CG_REQUIRE(
        stack.Execute(inputs, destination, settings, view, 1.0f / 60.0f, nullptr, &timings).IsOk());
    CG_EXPECT_TRUE(timings.passes[static_cast<i32>(PostProcessPass::AmbientOcclusion)].executed);

    const f32 crease = stack.AmbientOcclusionBuffer().GetPixel(kSize / 2 + 1, kSize / 2).r;
    const f32 flat = stack.AmbientOcclusionBuffer().GetPixel(kSize - 3, kSize / 2).r;
    // The far side of the step is occluded by the near block; the wall far
    // from the step is not.
    CG_EXPECT_TRUE(crease < 0.95f);
    CG_EXPECT_TRUE(flat > 0.98f);
    CG_EXPECT_TRUE(Luminance(destination.GetPixel(kSize / 2 + 1, kSize / 2)) <
                   Luminance(destination.GetPixel(kSize - 3, kSize / 2)));
}

CG_TEST(DepthOfFieldBlursTheBackgroundAndKeepsTheFocalPlaneSharp)
{
    constexpr i32 kSize = 64;
    constexpr f32 kNear = 0.1f;
    constexpr f32 kFar = 200.0f;

    HeapAllocator heap;
    auto created = PostProcessStack::Create(kSize, kSize, heap);
    CG_REQUIRE(created.IsOk());
    PostProcessStack stack = std::move(created).Value();

    Framebuffer color;
    Framebuffer destination;
    DepthBuffer depth;
    CG_REQUIRE(MakeImage(heap, color, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, destination, kSize, kSize));
    CG_REQUIRE(MakeDepth(heap, depth, kSize, kSize));

    // Top half: a sharp checker at the focus distance. Bottom half: the same
    // checker far behind it.
    for (i32 y = 0; y < kSize; ++y)
    {
        const bool background = y >= kSize / 2;
        const f32 distance = background ? 60.0f : 2.0f;
        for (i32 x = 0; x < kSize; ++x)
        {
            const bool white = ((x / 4) % 2) == 0;
            color.SetPixel(x, y, white ? Color{1.0f, 1.0f, 1.0f, 1.0f} : Color{0.0f, 0.0f, 0.0f, 1.0f});
            depth.TestAndWrite(x, y, DepthForDistance(distance, kNear, kFar));
        }
    }

    PostProcessInputs inputs{};
    inputs.color = &color;
    inputs.depth = &depth;

    PostProcessSettings settings{};
    settings.enableAmbientOcclusion = false;
    settings.enableBloom = false;
    settings.enableExposure = false;
    settings.enableToneMap = false;
    settings.enableDepthOfField = true;
    settings.depthOfField.focusDistance = 2.0f;
    settings.depthOfField.focalLengthMm = 85.0f;
    settings.depthOfField.apertureFStop = 1.4f;
    settings.depthOfField.maxCocPixels = 10.0f;

    PostProcessView view{};
    view.projection = Mat4::PerspectiveReverseZ(1.0471975512f, 1.0f, kNear, kFar);
    view.nearPlane = kNear;
    view.farPlane = kFar;

    PostProcessTimings timings{};
    CG_REQUIRE(
        stack.Execute(inputs, destination, settings, view, 1.0f / 60.0f, nullptr, &timings).IsOk());
    CG_EXPECT_TRUE(timings.passes[static_cast<i32>(PostProcessPass::DepthOfField)].executed);

    // Contrast across one checker edge, in focus and behind it.
    const f32 sharpStep = std::abs(Luminance(destination.GetPixel(3, 8)) -
                                   Luminance(destination.GetPixel(4, 8)));
    const f32 blurredStep = std::abs(Luminance(destination.GetPixel(3, kSize - 8)) -
                                     Luminance(destination.GetPixel(4, kSize - 8)));
    CG_EXPECT_TRUE(sharpStep > 0.9f);
    CG_EXPECT_TRUE(blurredStep < sharpStep * 0.5f);
}

CG_TEST(MotionBlurFollowsTheVelocityBuffer)
{
    constexpr i32 kSize = 64;

    HeapAllocator heap;
    auto created = PostProcessStack::Create(kSize, kSize, heap);
    CG_REQUIRE(created.IsOk());
    PostProcessStack stack = std::move(created).Value();

    Framebuffer color;
    Framebuffer velocity;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, color, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, velocity, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, destination, kSize, kSize));

    color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    color.ClearRect(30, 0, 4, kSize, Color{1.0f, 1.0f, 1.0f, 1.0f});
    // Twenty pixels per frame to the right, everywhere.
    velocity.Clear(Color{20.0f, 0.0f, 0.0f, 1.0f});

    PostProcessInputs inputs{};
    inputs.color = &color;
    inputs.velocity = &velocity;

    PostProcessSettings settings{};
    settings.enableAmbientOcclusion = false;
    settings.enableBloom = false;
    settings.enableExposure = false;
    settings.enableToneMap = false;
    settings.enableMotionBlur = true;
    settings.motionBlur.shutterAngleDegrees = 360.0f;
    settings.motionBlur.sampleCount = 16;

    PostProcessTimings timings{};
    CG_REQUIRE(stack
                   .Execute(inputs, destination, settings, MakeView(), 1.0f / 60.0f, nullptr,
                            &timings)
                   .IsOk());
    CG_EXPECT_TRUE(timings.passes[static_cast<i32>(PostProcessPass::MotionBlur)].executed);

    // The bar smears along x and not along y.
    CG_EXPECT_TRUE(Luminance(destination.GetPixel(24, 32)) > 0.05f);
    CG_EXPECT_TRUE(Luminance(destination.GetPixel(40, 32)) > 0.05f);
    CG_EXPECT_TRUE(Luminance(destination.GetPixel(32, 32)) < 1.0f);

    // A pixel far outside the shutter reach is untouched.
    CG_EXPECT_NEAR(Luminance(destination.GetPixel(2, 32)), 0.0f, 1.0e-6f);
}

CG_TEST(ChainIsDeterministicWithAndWithoutJobs)
{
    constexpr i32 kSize = 64;
    constexpr f32 kNear = 0.1f;
    constexpr f32 kFar = 100.0f;

    HeapAllocator heap;
    auto jobsCreated = cyber::jobs::JobSystem::Create(cyber::jobs::JobSystem::Desc{4, nullptr}, heap);
    CG_REQUIRE(jobsCreated.IsOk());
    cyber::jobs::JobSystem jobSystem = std::move(jobsCreated).Value();

    Framebuffer color;
    Framebuffer normals;
    Framebuffer velocity;
    Framebuffer serial;
    Framebuffer parallel;
    DepthBuffer depth;
    CG_REQUIRE(MakeImage(heap, color, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, normals, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, velocity, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, serial, kSize, kSize));
    CG_REQUIRE(MakeImage(heap, parallel, kSize, kSize));
    CG_REQUIRE(MakeDepth(heap, depth, kSize, kSize));

    for (i32 y = 0; y < kSize; ++y)
    {
        for (i32 x = 0; x < kSize; ++x)
        {
            const f32 wave = 0.5f + 0.5f * std::sin(static_cast<f32>(x) * 0.3f) *
                                        std::cos(static_cast<f32>(y) * 0.21f);
            color.SetPixel(x, y, Color{wave * 3.0f, wave * 2.0f, wave, 1.0f});
            normals.SetPixel(x, y, Color{0.5f, 0.5f, 0.0f, 1.0f});
            velocity.SetPixel(x, y, Color{6.0f, -3.0f, 0.0f, 1.0f});
            depth.TestAndWrite(x, y, DepthForDistance(2.0f + wave * 6.0f, kNear, kFar));
        }
    }

    auto gradeCreated = ColorGradeLut::Bake(ColorGradeSettings{}, 24, heap);
    CG_REQUIRE(gradeCreated.IsOk());
    const ColorGradeLut lut = std::move(gradeCreated).Value();

    PostProcessInputs inputs{};
    inputs.color = &color;
    inputs.depth = &depth;
    inputs.normals = &normals;
    inputs.velocity = &velocity;

    PostProcessSettings settings{};
    settings.enableAmbientOcclusion = true;
    settings.enableDepthOfField = true;
    settings.enableMotionBlur = true;
    settings.enableBloom = true;
    settings.enableExposure = true;
    settings.enableToneMap = true;
    settings.enableColorGrade = true;
    settings.colorGradeLut = &lut;

    PostProcessView view{};
    view.projection = Mat4::PerspectiveReverseZ(1.0471975512f, 1.0f, kNear, kFar);
    view.nearPlane = kNear;
    view.farPlane = kFar;

    auto serialStack = PostProcessStack::Create(kSize, kSize, heap);
    CG_REQUIRE(serialStack.IsOk());
    PostProcessStack serialChain = std::move(serialStack).Value();
    settings.useJobs = false;
    CG_REQUIRE(
        serialChain.Execute(inputs, serial, settings, view, 1.0f / 60.0f, nullptr, nullptr).IsOk());

    auto parallelStack = PostProcessStack::Create(kSize, kSize, heap);
    CG_REQUIRE(parallelStack.IsOk());
    PostProcessStack parallelChain = std::move(parallelStack).Value();
    settings.useJobs = true;
    CG_REQUIRE(parallelChain
                   .Execute(inputs, parallel, settings, view, 1.0f / 60.0f, &jobSystem, nullptr)
                   .IsOk());

    CG_EXPECT_TRUE(ImagesEqual(serial, parallel));
}

CG_TEST(ExposureAdaptationIsCarriedBetweenFrames)
{
    HeapAllocator heap;
    auto created = PostProcessStack::Create(32, 32, heap);
    CG_REQUIRE(created.IsOk());
    PostProcessStack stack = std::move(created).Value();

    Framebuffer color;
    Framebuffer destination;
    CG_REQUIRE(MakeImage(heap, color, 32, 32));
    CG_REQUIRE(MakeImage(heap, destination, 32, 32));

    PostProcessInputs inputs{};
    inputs.color = &color;
    PostProcessSettings settings{};
    settings.enableAmbientOcclusion = false;
    settings.enableBloom = false;
    settings.exposure.adaptUpEvPerSecond = 1.0f;
    settings.exposure.adaptDownEvPerSecond = 1.0f;

    // Frame one: a dark scene, which snaps.
    color.Clear(Color{0.02f, 0.02f, 0.02f, 1.0f});
    PostProcessTimings first{};
    CG_REQUIRE(stack
                   .Execute(inputs, destination, settings, MakeView(), 1.0f / 60.0f, nullptr,
                            &first)
                   .IsOk());
    const f32 darkEv = first.exposureEv;
    CG_EXPECT_TRUE(darkEv > 1.0f);

    // Frame two: the lights come on. One sixtieth of a second at 1 EV/s moves
    // the camera a sixtieth of a stop, not the whole way.
    color.Clear(Color{4.0f, 4.0f, 4.0f, 1.0f});
    PostProcessTimings second{};
    CG_REQUIRE(stack
                   .Execute(inputs, destination, settings, MakeView(), 1.0f / 60.0f, nullptr,
                            &second)
                   .IsOk());
    CG_EXPECT_NEAR(second.exposureEv, darkEv - 1.0f / 60.0f, 1.0e-3f);

    // The image darkens as the camera stops down over the following seconds.
    // The scene is six stops above middle grey, so this takes seconds, not
    // frames, which is the point of an adaptation rate.
    const f32 previousMean = MeanLuminance(destination);
    for (i32 frame = 0; frame < 600; ++frame)
    {
        CG_REQUIRE(stack
                       .Execute(inputs, destination, settings, MakeView(), 1.0f / 60.0f, nullptr,
                                nullptr)
                       .IsOk());
    }
    const f32 settledMean = MeanLuminance(destination);
    CG_EXPECT_TRUE(settledMean < previousMean);
    CG_EXPECT_TRUE(stack.Exposure().currentEv < darkEv);
}
