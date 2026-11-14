#include "gi_test_scenes.h"

#include <cyber/memory/heap_allocator.h>
#include <cyber/software/gi_quality.h>
#include <cyber/test/test.h>

#include <cmath>
#include <limits>
#include <vector>

using namespace cyber::software;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::math::Color;
using cyber::math::Vec3;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using cyber::software::test::MakeImage;

CG_TEST(software_gi_quality_irradiance_error)
{
    const std::vector<Vec3> reference = {
        {1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 0.0f}};
    IrradianceErrorMetrics metrics{};
    CG_REQUIRE(MeasureIrradianceError(Span<const Vec3>(reference.data(), reference.size()),
                                      Span<const Vec3>(reference.data(), reference.size()), 0.01f,
                                      0.1f, metrics)
                   .IsOk());
    CG_EXPECT_EQ(metrics.sampleCount, u32{4});
    CG_EXPECT_NEAR(metrics.meanRelativeError, 0.0f, 1.0e-7f);
    CG_EXPECT_NEAR(metrics.energyRatio, 1.0f, 1.0e-6f);
    CG_EXPECT_EQ(metrics.failingSamples, u32{0});

    // A uniformly 20% dark estimate: error and bias both 0.2, energy 0.8.
    std::vector<Vec3> dark;
    for (const Vec3& value : reference)
    {
        dark.push_back(value * 0.8f);
    }
    CG_REQUIRE(MeasureIrradianceError(Span<const Vec3>(dark.data(), dark.size()),
                                      Span<const Vec3>(reference.data(), reference.size()), 0.01f,
                                      0.1f, metrics)
                   .IsOk());
    // The black reference sample has zero error (0.8 * 0 = 0), so three of
    // four samples carry 0.2.
    CG_EXPECT_NEAR(metrics.meanRelativeError, 0.15f, 1.0e-5f);
    CG_EXPECT_NEAR(metrics.meanRelativeBias, -0.15f, 1.0e-5f);
    CG_EXPECT_NEAR(metrics.maxRelativeError, 0.2f, 1.0e-5f);
    CG_EXPECT_NEAR(metrics.energyRatio, 0.8f, 1.0e-5f);
    CG_EXPECT_EQ(metrics.failingSamples, u32{3});
    CG_EXPECT_NEAR(metrics.failingFraction, 0.75f, 1.0e-6f);

    // The floor bounds the error of a small absolute miss on a black
    // reference.
    const std::vector<Vec3> nearlyBlack = {{0.001f, 0.001f, 0.001f}};
    const std::vector<Vec3> black = {{0.0f, 0.0f, 0.0f}};
    CG_REQUIRE(MeasureIrradianceError(Span<const Vec3>(nearlyBlack.data(), 1),
                                      Span<const Vec3>(black.data(), 1), 0.01f, 0.5f, metrics)
                   .IsOk());
    CG_EXPECT_NEAR(metrics.maxRelativeError, 0.1f, 1.0e-4f);

    // Validation.
    CG_EXPECT_TRUE(
        MeasureIrradianceError(Span<const Vec3>(), Span<const Vec3>(), 0.01f, 0.1f, metrics)
            .IsError());
    CG_EXPECT_TRUE(MeasureIrradianceError(Span<const Vec3>(dark.data(), 2),
                                          Span<const Vec3>(reference.data(), 3), 0.01f, 0.1f,
                                          metrics)
                       .IsError());
    CG_EXPECT_TRUE(MeasureIrradianceError(Span<const Vec3>(dark.data(), 2),
                                          Span<const Vec3>(reference.data(), 2), 0.0f, 0.1f,
                                          metrics)
                       .IsError());
    std::vector<Vec3> poisoned = dark;
    poisoned[1].y = std::numeric_limits<f32>::quiet_NaN();
    CG_EXPECT_TRUE(MeasureIrradianceError(Span<const Vec3>(poisoned.data(), 4),
                                          Span<const Vec3>(reference.data(), 4), 0.01f, 0.1f,
                                          metrics)
                       .IsError());
}

CG_TEST(software_gi_quality_light_leak)
{
    const std::vector<Vec3> sealed = {{0.0f, 0.0f, 0.0f}, {0.001f, 0.001f, 0.001f}};
    const std::vector<Vec3> leaking = {{0.0f, 0.0f, 0.0f}, {0.3f, 0.3f, 0.3f}, {0.5f, 0.5f, 0.5f}};
    LightLeakMetrics metrics{};
    CG_REQUIRE(MeasureLightLeak(Span<const Vec3>(sealed.data(), sealed.size()),
                                Vec3{1.0f, 1.0f, 1.0f}, 0.05f, metrics)
                   .IsOk());
    CG_EXPECT_EQ(metrics.leakingSamples, u32{0});
    CG_EXPECT_NEAR(metrics.maxLeakRatio, 0.001f, 1.0e-6f);
    CG_REQUIRE(MeasureLightLeak(Span<const Vec3>(leaking.data(), leaking.size()),
                                Vec3{1.0f, 1.0f, 1.0f}, 0.05f, metrics)
                   .IsOk());
    CG_EXPECT_EQ(metrics.leakingSamples, u32{2});
    CG_EXPECT_NEAR(metrics.meanLeakRatio, 0.8f / 3.0f, 1.0e-5f);
    CG_EXPECT_NEAR(metrics.maxLeakRatio, 0.5f, 1.0e-6f);

    CG_EXPECT_TRUE(
        MeasureLightLeak(Span<const Vec3>(), Vec3{1.0f, 1.0f, 1.0f}, 0.05f, metrics).IsError());
    CG_EXPECT_TRUE(
        MeasureLightLeak(Span<const Vec3>(sealed.data(), 1), Vec3{}, 0.05f, metrics).IsError());
    CG_EXPECT_TRUE(
        MeasureLightLeak(Span<const Vec3>(sealed.data(), 1), Vec3{1.0f, 1.0f, 1.0f}, 0.0f, metrics)
            .IsError());
}

// Noise and bias are separate numbers: a noisy unbiased image fails one, a
// smooth darkened image fails the other.
CG_TEST(software_gi_quality_noise_and_bias)
{
    HeapAllocator allocator;
    constexpr i32 kSize = 32;
    Framebuffer reference = MakeImage(allocator, kSize, kSize);
    Framebuffer noisy = MakeImage(allocator, kSize, kSize);
    Framebuffer dark = MakeImage(allocator, kSize, kSize);
    CG_REQUIRE(reference.IsValid() && noisy.IsValid() && dark.IsValid());
    reference.Clear(Color{0.5f, 0.5f, 0.5f, 1.0f});
    dark.Clear(Color{0.4f, 0.4f, 0.4f, 1.0f});
    for (i32 y = 0; y < kSize; ++y)
    {
        for (i32 x = 0; x < kSize; ++x)
        {
            // Zero-mean checkerboard noise of +-0.2.
            const f32 noise = ((x + y) % 2 == 0) ? 0.2f : -0.2f;
            noisy.SetPixel(x, y, Color{0.5f + noise, 0.5f + noise, 0.5f + noise, 1.0f});
        }
    }

    NoiseMetrics metrics{};
    CG_REQUIRE(MeasureNoise(reference, reference, metrics).IsOk());
    CG_EXPECT_NEAR(metrics.rmse, 0.0f, 1.0e-7f);
    CG_EXPECT_NEAR(metrics.laplacianEnergy, 0.0f, 1.0e-7f);
    CG_EXPECT_NEAR(metrics.referenceMeanLuminance, 0.5f, 1.0e-5f);

    CG_REQUIRE(MeasureNoise(noisy, reference, metrics).IsOk());
    CG_EXPECT_NEAR(metrics.rmse, 0.2f, 1.0e-4f);
    CG_EXPECT_NEAR(metrics.relativeRmse, 0.4f, 1.0e-4f);
    CG_EXPECT_NEAR(metrics.bias, 0.0f, 1.0e-4f);
    // A checkerboard is the worst case for the Laplacian: 8 times the
    // amplitude.
    CG_EXPECT_NEAR(metrics.laplacianEnergy, 1.6f, 1.0e-3f);

    CG_REQUIRE(MeasureNoise(dark, reference, metrics).IsOk());
    CG_EXPECT_NEAR(metrics.rmse, 0.1f, 1.0e-4f);
    CG_EXPECT_NEAR(metrics.relativeBias, -0.2f, 1.0e-4f);
    CG_EXPECT_NEAR(metrics.laplacianEnergy, 0.0f, 1.0e-6f);

    f32 energy = 0.0f;
    CG_REQUIRE(MeasureLaplacianEnergy(noisy, energy).IsOk());
    CG_EXPECT_NEAR(energy, 1.6f, 1.0e-3f);
    Framebuffer tiny = MakeImage(allocator, 2, 2);
    CG_EXPECT_TRUE(MeasureLaplacianEnergy(tiny, energy).IsError());
    Framebuffer black = MakeImage(allocator, kSize, kSize);
    black.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    CG_EXPECT_TRUE(MeasureNoise(noisy, black, metrics).IsError());
    CG_EXPECT_TRUE(MeasureNoise(noisy, tiny, metrics).IsError());
}
