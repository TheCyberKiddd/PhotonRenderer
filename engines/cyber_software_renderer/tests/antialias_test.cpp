#include <cyber/jobs/job_system.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/software/antialias.h>
#include <cyber/software/framebuffer.h>
#include <cyber/test/test.h>

#include <cmath>
#include <utility>

using namespace cyber::software;
using namespace cyber::memory;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::core::usize;
using cyber::math::Color;

namespace
{

constexpr ResolveFilter kAllFilters[4] = {
    ResolveFilter::Box,
    ResolveFilter::Tent,
    ResolveFilter::Mitchell,
    ResolveFilter::BlackmanHarris,
};

[[nodiscard]] bool MakeTarget(HeapAllocator& heap, Framebuffer& out, i32 width, i32 height,
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

// Fills a supersampled source with a half-plane split by a diagonal edge, so
// the resolve has partial coverage to reconstruct.
void FillDiagonalEdge(Framebuffer& source, i32 factor)
{
    const f32 scale = 1.0f / static_cast<f32>(factor);
    for (i32 y = 0; y < source.Height(); ++y)
    {
        for (i32 x = 0; x < source.Width(); ++x)
        {
            const f32 u = (static_cast<f32>(x) + 0.5f) * scale;
            const f32 v = (static_cast<f32>(y) + 0.5f) * scale;
            const bool inside = u * 0.37f + v > 6.0f;
            source.SetPixel(x, y,
                            inside ? Color{1.0f, 1.0f, 1.0f, 1.0f} : Color{0.0f, 0.0f, 0.0f, 1.0f});
        }
    }
}

// Number of distinct quantized luminance steps along a column, which is how
// many gradations the filter produced across the edge.
[[nodiscard]] u32 CountGrayLevels(const Framebuffer& image)
{
    bool seen[256]{};
    for (i32 y = 0; y < image.Height(); ++y)
    {
        for (i32 x = 0; x < image.Width(); ++x)
        {
            seen[EncodeSrgb8(image.GetPixel(x, y).r)] = true;
        }
    }
    u32 count = 0;
    for (bool entry : seen)
    {
        count += entry ? 1u : 0u;
    }
    return count;
}

[[nodiscard]] bool PixelsEqual(const Framebuffer& lhs, const Framebuffer& rhs)
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

}  // namespace

CG_TEST(antialias_quality_presets_are_sample_rates)
{
    static_assert(SupersampleFactorForQuality(AntiAliasQuality::Off) == 1);
    static_assert(SupersampleFactorForQuality(AntiAliasQuality::Balanced) == 2);
    static_assert(SupersampleFactorForQuality(AntiAliasQuality::High) == 4);
    static_assert(SupersampleFactorForQuality(AntiAliasQuality::Ultra) == 8);
    static_assert(SampleCountForQuality(AntiAliasQuality::High) == 16);
    static_assert(SampleCountForQuality(AntiAliasQuality::Ultra) == 64);
    static_assert(SupersampleFactorForQuality(static_cast<AntiAliasQuality>(99)) ==
                  kDefaultSupersampleFactor);

    // The renderer-wide default is 4x4 Mitchell.
    static_assert(kDefaultAntiAliasQuality == AntiAliasQuality::High);
    static_assert(kDefaultSupersampleFactor == 4);
    static_assert(kDefaultResolveFilter == ResolveFilter::Mitchell);

    static_assert(ResolveFilterRadius(ResolveFilter::Box) == 0.5f);
    static_assert(ResolveFilterRadius(ResolveFilter::Tent) == 1.0f);
    static_assert(ResolveFilterRadius(ResolveFilter::Mitchell) == 2.0f);
}

CG_TEST(antialias_filters_are_normalized_and_bounded)
{
    for (ResolveFilter filter : kAllFilters)
    {
        const f32 radius = ResolveFilterRadius(filter);

        // Zero outside the support, on both sides.
        CG_EXPECT_EQ(EvaluateResolveFilter(filter, radius + 0.01f), 0.0f);
        CG_EXPECT_EQ(EvaluateResolveFilter(filter, -(radius + 0.01f)), 0.0f);
        // Even symmetry.
        CG_EXPECT_NEAR(EvaluateResolveFilter(filter, 0.31f), EvaluateResolveFilter(filter, -0.31f),
                       1e-6);
        // Positive at the center and non-finite input is rejected.
        CG_EXPECT_TRUE(EvaluateResolveFilter(filter, 0.0f) > 0.0f);
        CG_EXPECT_EQ(EvaluateResolveFilter(filter, std::nanf("")), 0.0f);

        // Riemann sum of the kernel over its support, which must be the unit
        // area a reconstruction filter needs to preserve brightness.
        constexpr i32 kSteps = 20000;
        const f32 step = (2.0f * radius) / static_cast<f32>(kSteps);
        f32 area = 0.0f;
        for (i32 index = 0; index < kSteps; ++index)
        {
            const f32 offset = -radius + (static_cast<f32>(index) + 0.5f) * step;
            area += EvaluateResolveFilter(filter, offset) * step;
        }
        CG_EXPECT_NEAR(area, 1.0f, 0.02);
    }

    // Mitchell has to undershoot outside its first lobe; that negative lobe is
    // what keeps a resolved edge from reading as blurred.
    CG_EXPECT_TRUE(EvaluateResolveFilter(ResolveFilter::Mitchell, 1.5f) < 0.0f);
    // Box and tent never do.
    CG_EXPECT_TRUE(EvaluateResolveFilter(ResolveFilter::Box, 0.4f) > 0.0f);
    CG_EXPECT_TRUE(EvaluateResolveFilter(ResolveFilter::Tent, 0.9f) > 0.0f);
}

CG_TEST(antialias_resolve_preserves_a_constant_image)
{
    HeapAllocator heap;
    constexpr i32 kWidth = 23;
    constexpr i32 kHeight = 17;
    const Color constant{0.25f, 0.5f, 0.75f, 0.5f};

    for (ResolveFilter filter : kAllFilters)
    {
        for (i32 factor : {1, 2, 3, 4, 8})
        {
            Framebuffer source;
            Framebuffer resolved;
            CG_REQUIRE(MakeTarget(heap, source, kWidth * factor, kHeight * factor));
            CG_REQUIRE(MakeTarget(heap, resolved, kWidth, kHeight));
            source.Clear(constant);
            resolved.Clear(Color{1.0f, 0.0f, 0.0f, 1.0f});

            ResolveDesc desc{};
            desc.factor = factor;
            desc.filter = filter;
            CG_REQUIRE(ResolveSupersampled(source, resolved, desc, nullptr, heap).IsOk());

            // Edge clamping plus normalized weights means every pixel,
            // including the border, reproduces the constant exactly.
            for (i32 y = 0; y < kHeight; ++y)
            {
                for (i32 x = 0; x < kWidth; ++x)
                {
                    const Color pixel = resolved.GetPixel(x, y);
                    CG_EXPECT_NEAR(pixel.r, constant.r, 1e-5);
                    CG_EXPECT_NEAR(pixel.g, constant.g, 1e-5);
                    CG_EXPECT_NEAR(pixel.b, constant.b, 1e-5);
                    CG_EXPECT_NEAR(pixel.a, constant.a, 1e-5);
                }
            }
        }
    }
}

CG_TEST(antialias_box_resolve_matches_the_sample_average)
{
    HeapAllocator heap;
    constexpr i32 kFactor = 4;
    constexpr i32 kWidth = 9;
    constexpr i32 kHeight = 7;

    Framebuffer source;
    Framebuffer resolved;
    CG_REQUIRE(MakeTarget(heap, source, kWidth * kFactor, kHeight * kFactor));
    CG_REQUIRE(MakeTarget(heap, resolved, kWidth, kHeight));
    FillDiagonalEdge(source, kFactor);

    ResolveDesc desc{};
    desc.factor = kFactor;
    desc.filter = ResolveFilter::Box;
    CG_REQUIRE(ResolveSupersampled(source, resolved, desc, nullptr, heap).IsOk());

    const f32 inverse = 1.0f / static_cast<f32>(kFactor * kFactor);
    for (i32 y = 0; y < kHeight; ++y)
    {
        for (i32 x = 0; x < kWidth; ++x)
        {
            f32 sum = 0.0f;
            for (i32 sampleY = 0; sampleY < kFactor; ++sampleY)
            {
                for (i32 sampleX = 0; sampleX < kFactor; ++sampleX)
                {
                    sum += source.GetPixel(x * kFactor + sampleX, y * kFactor + sampleY).r;
                }
            }
            CG_EXPECT_NEAR(resolved.GetPixel(x, y).r, sum * inverse, 1e-5);
        }
    }
}

CG_TEST(antialias_resolve_smooths_an_edge)
{
    HeapAllocator heap;
    constexpr i32 kWidth = 48;
    constexpr i32 kHeight = 32;
    constexpr i32 kFactor = 4;

    Framebuffer aliasedSource;
    Framebuffer aliased;
    CG_REQUIRE(MakeTarget(heap, aliasedSource, kWidth, kHeight));
    CG_REQUIRE(MakeTarget(heap, aliased, kWidth, kHeight));
    FillDiagonalEdge(aliasedSource, 1);
    {
        ResolveDesc desc{};
        desc.factor = 1;
        desc.filter = ResolveFilter::Box;
        CG_REQUIRE(ResolveSupersampled(aliasedSource, aliased, desc, nullptr, heap).IsOk());
    }
    // One sample per pixel and a box filter is exactly the un-anti-aliased
    // image: pure black and pure white, nothing between.
    CG_EXPECT_EQ(CountGrayLevels(aliased), u32{2});

    Framebuffer source;
    CG_REQUIRE(MakeTarget(heap, source, kWidth * kFactor, kHeight * kFactor));
    FillDiagonalEdge(source, kFactor);

    for (ResolveFilter filter : kAllFilters)
    {
        Framebuffer resolved;
        CG_REQUIRE(MakeTarget(heap, resolved, kWidth, kHeight));
        ResolveDesc desc{};
        desc.factor = kFactor;
        desc.filter = filter;
        CG_REQUIRE(ResolveSupersampled(source, resolved, desc, nullptr, heap).IsOk());

        // Every filter has to produce a ramp, not a staircase.
        CG_EXPECT_TRUE(CountGrayLevels(resolved) > 8);

        // No channel may leave [0, 1] far enough to clip visibly: the
        // negative-lobe filters are clamped below and the window keeps the
        // overshoot small above.
        for (i32 y = 0; y < kHeight; ++y)
        {
            for (i32 x = 0; x < kWidth; ++x)
            {
                const Color pixel = resolved.GetPixel(x, y);
                CG_EXPECT_TRUE(pixel.r >= 0.0f);
                CG_EXPECT_TRUE(pixel.r <= 1.1f);
            }
        }
    }

    // The wider the support, the more gradations across the same edge.
    Framebuffer boxResolved;
    Framebuffer mitchellResolved;
    CG_REQUIRE(MakeTarget(heap, boxResolved, kWidth, kHeight));
    CG_REQUIRE(MakeTarget(heap, mitchellResolved, kWidth, kHeight));
    ResolveDesc boxDesc{};
    boxDesc.factor = kFactor;
    boxDesc.filter = ResolveFilter::Box;
    ResolveDesc mitchellDesc{};
    mitchellDesc.factor = kFactor;
    mitchellDesc.filter = ResolveFilter::Mitchell;
    CG_REQUIRE(ResolveSupersampled(source, boxResolved, boxDesc, nullptr, heap).IsOk());
    CG_REQUIRE(ResolveSupersampled(source, mitchellResolved, mitchellDesc, nullptr, heap).IsOk());
    CG_EXPECT_TRUE(CountGrayLevels(mitchellResolved) >= CountGrayLevels(boxResolved));
}

CG_TEST(antialias_resolve_is_deterministic_across_scheduling)
{
    HeapAllocator heap;
    constexpr i32 kWidth = 37;
    constexpr i32 kHeight = 29;
    constexpr i32 kFactor = 4;

    Framebuffer source;
    CG_REQUIRE(MakeTarget(heap, source, kWidth * kFactor, kHeight * kFactor));
    FillDiagonalEdge(source, kFactor);

    Framebuffer serial;
    CG_REQUIRE(MakeTarget(heap, serial, kWidth, kHeight));
    ResolveDesc serialDesc{};
    serialDesc.factor = kFactor;
    serialDesc.useJobs = false;
    CG_REQUIRE(ResolveSupersampled(source, serial, serialDesc, nullptr, heap).IsOk());

    // Band size only changes the temporary buffer, never the image.
    for (i32 bandRows : {1, 3, 8, 64, 4096})
    {
        Framebuffer banded;
        CG_REQUIRE(MakeTarget(heap, banded, kWidth, kHeight));
        ResolveDesc desc = serialDesc;
        desc.bandRows = bandRows;
        CG_REQUIRE(ResolveSupersampled(source, banded, desc, nullptr, heap).IsOk());
        CG_EXPECT_TRUE(PixelsEqual(serial, banded));
    }

    auto created = cyber::jobs::JobSystem::Create(cyber::jobs::JobSystem::Desc{4, nullptr}, heap);
    CG_REQUIRE(created.IsOk());
    cyber::jobs::JobSystem jobSystem = std::move(created).Value();

    for (i32 bandRows : {1, 5, 32})
    {
        Framebuffer parallel;
        CG_REQUIRE(MakeTarget(heap, parallel, kWidth, kHeight));
        ResolveDesc desc{};
        desc.factor = kFactor;
        desc.bandRows = bandRows;
        desc.useJobs = true;
        CG_REQUIRE(ResolveSupersampled(source, parallel, desc, &jobSystem, heap).IsOk());
        CG_EXPECT_TRUE(PixelsEqual(serial, parallel));
    }
    jobSystem.Shutdown();
}

CG_TEST(antialias_resolve_rejects_invalid_input)
{
    HeapAllocator heap;
    Framebuffer source;
    Framebuffer resolved;
    CG_REQUIRE(MakeTarget(heap, source, 16, 16));
    CG_REQUIRE(MakeTarget(heap, resolved, 4, 4));

    Framebuffer invalid;
    ResolveDesc desc{};
    desc.factor = 4;
    CG_EXPECT_TRUE(ResolveSupersampled(invalid, resolved, desc, nullptr, heap).IsError());
    CG_EXPECT_TRUE(ResolveSupersampled(source, invalid, desc, nullptr, heap).IsError());

    desc.factor = 0;
    CG_EXPECT_TRUE(ResolveSupersampled(source, resolved, desc, nullptr, heap).IsError());
    desc.factor = kMaxSupersampleFactor + 1;
    CG_EXPECT_TRUE(ResolveSupersampled(source, resolved, desc, nullptr, heap).IsError());

    // 16 is not 4 x 2.
    desc.factor = 2;
    CG_EXPECT_TRUE(ResolveSupersampled(source, resolved, desc, nullptr, heap).IsError());

    desc.factor = 1;
    CG_EXPECT_TRUE(ResolveSupersampled(source, source, desc, nullptr, heap).IsError());
}

CG_TEST(antialias_target_sizes_and_resolves)
{
    HeapAllocator heap;
    AntiAliasTargetDesc desc{};
    desc.width = 32;
    desc.height = 18;

    auto created = AntiAliasTarget::Create(desc, heap);
    CG_REQUIRE(created.IsOk());
    AntiAliasTarget target = std::move(created).Value();

    CG_EXPECT_TRUE(target.IsValid());
    CG_EXPECT_EQ(target.Factor(), i32{4});
    CG_EXPECT_EQ(target.SampleCount(), i32{16});
    CG_EXPECT_EQ(target.Width(), i32{32});
    CG_EXPECT_EQ(target.Height(), i32{18});
    CG_EXPECT_EQ(target.RenderWidth(), i32{128});
    CG_EXPECT_EQ(target.RenderHeight(), i32{72});
    CG_EXPECT_EQ(target.Color().Width(), i32{128});
    CG_EXPECT_EQ(target.Depth().Height(), i32{72});
    CG_EXPECT_TRUE(target.Filter() == ResolveFilter::Mitchell);

    const Viewport viewport = target.RenderViewport();
    CG_EXPECT_EQ(viewport.width, 128.0f);
    CG_EXPECT_EQ(viewport.height, 72.0f);

    // Clear sets the color target and resets depth to the reverse-Z far value.
    target.Depth().TestAndWrite(0, 0, 0.9f);
    target.Clear(Color{0.1f, 0.2f, 0.3f, 1.0f});
    CG_EXPECT_EQ(target.Depth().Get(0, 0), 0.0f);

    Framebuffer resolved;
    CG_REQUIRE(MakeTarget(heap, resolved, 32, 18, PixelFormat::RGBA8Unorm));
    CG_REQUIRE(target.Resolve(resolved, nullptr, heap).IsOk());
    const Color pixel = resolved.GetPixel(5, 5);
    CG_EXPECT_NEAR(pixel.r, 0.1f, 0.01);
    CG_EXPECT_NEAR(pixel.g, 0.2f, 0.01);
    CG_EXPECT_NEAR(pixel.b, 0.3f, 0.01);

    // A destination that is not the resolved size is rejected.
    Framebuffer wrong;
    CG_REQUIRE(MakeTarget(heap, wrong, 31, 18));
    CG_EXPECT_TRUE(target.Resolve(wrong, nullptr, heap).IsError());
}

CG_TEST(antialias_target_rejects_invalid_descriptors)
{
    HeapAllocator heap;

    AntiAliasTargetDesc zero{};
    CG_EXPECT_TRUE(AntiAliasTarget::Create(zero, heap).IsError());

    AntiAliasTargetDesc badFactor{};
    badFactor.width = 8;
    badFactor.height = 8;
    badFactor.factor = kMaxSupersampleFactor + 1;
    CG_EXPECT_TRUE(AntiAliasTarget::Create(badFactor, heap).IsError());

    // Supersampling must not push the raster target past the framebuffer limit.
    AntiAliasTargetDesc tooLarge{};
    tooLarge.width = 4096;
    tooLarge.height = 4096;
    tooLarge.quality = AntiAliasQuality::Ultra;
    CG_EXPECT_TRUE(AntiAliasTarget::Create(tooLarge, heap).IsError());

    AntiAliasTargetDesc badFormat{};
    badFormat.width = 8;
    badFormat.height = 8;
    badFormat.format = static_cast<PixelFormat>(42);
    CG_EXPECT_TRUE(AntiAliasTarget::Create(badFormat, heap).IsError());

    // Off is a valid quality: one sample per pixel and a usable target.
    AntiAliasTargetDesc off{};
    off.width = 8;
    off.height = 8;
    off.quality = AntiAliasQuality::Off;
    auto created = AntiAliasTarget::Create(off, heap);
    CG_REQUIRE(created.IsOk());
    AntiAliasTarget target = std::move(created).Value();
    CG_EXPECT_EQ(target.Factor(), i32{1});
    CG_EXPECT_EQ(target.RenderWidth(), i32{8});
}
