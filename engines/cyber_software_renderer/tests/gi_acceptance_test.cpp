// Phase 31 acceptance: the chosen GI method runs in the test scene, and the
// ray-traced signal it shares machinery with is usable once denoised.
//
// "Runs" is turned into three measured properties, each paired with a control
// configuration that fails the same threshold, because a threshold that
// nothing can fail measures nothing:
//
//   accuracy  The baked irradiance volume agrees with a path-traced ground
//             truth across a Cornell-style room. Control: a bake that carries
//             no interreflection (zero bounces) is too dark.
//   leakage   Light from a lit room does not reach a sealed room through the
//             shared wall the probe lattice straddles. Control: the same
//             volume sampled without its visibility and normal terms leaks.
//   noise     A one-sample-per-pixel ray-traced soft shadow, denoised, is
//             close to a converged reference and unbiased. Control: the raw
//             signal.
//
// Every number is printed, so the report quotes measurements rather than
// thresholds.

#include "gi_test_scenes.h"

#include <cyber/jobs/job_system.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/software/denoise.h>
#include <cyber/software/gi_quality.h>
#include <cyber/software/global_illumination.h>
#include <cyber/test/test.h>

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

using namespace cyber::software;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::math::Color;
using cyber::math::Vec3;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using cyber::software::test::Diffuse;
using cyber::software::test::EncodeNormal;
using cyber::software::test::GiTestScene;
using cyber::software::test::HashUniform;
using cyber::software::test::MakeDepth;
using cyber::software::test::MakeImage;
using cyber::software::test::ReverseZFromDistance;

namespace
{

constexpr f32 kPi = cyber::math::kPi;

struct Jobs
{
    cyber::jobs::JobSystem system;

    explicit Jobs(HeapAllocator& allocator)
    {
        cyber::jobs::JobSystem::Desc desc{};
        auto created = cyber::jobs::JobSystem::Create(desc, allocator);
        if (created.IsOk())
        {
            system = std::move(created).Value();
        }
    }

    ~Jobs()
    {
        system.Shutdown();
    }

    Jobs(const Jobs&) = delete;
    Jobs& operator=(const Jobs&) = delete;
};

IrradianceVolume MakeVolume(HeapAllocator& allocator, const Vec3& minimum, const Vec3& maximum,
                            u32 x, u32 y, u32 z)
{
    IrradianceVolumeDesc desc{};
    desc.boundsMin = minimum;
    desc.boundsMax = maximum;
    desc.probeCountX = x;
    desc.probeCountY = y;
    desc.probeCountZ = z;
    auto created = IrradianceVolume::Create(desc, allocator);
    return created.IsOk() ? std::move(created).Value() : IrradianceVolume{};
}

struct SamplePoint
{
    Vec3 position;
    Vec3 normal;
};

// Cornell-style room: 4 x 3 x 4, red wall at x = 0, green at x = 4, white
// elsewhere, a white block on the floor, and a lamp under the ceiling.
struct CornellRoom
{
    explicit CornellRoom(HeapAllocator& allocator) : builder(allocator) {}

    bool Build()
    {
        const u32 white = builder.AddGroup(Diffuse(0.75f, 0.75f, 0.75f));
        const u32 red = builder.AddGroup(Diffuse(0.75f, 0.08f, 0.08f));
        const u32 green = builder.AddGroup(Diffuse(0.08f, 0.75f, 0.08f));
        const u32 block = builder.AddGroup(Diffuse(0.75f, 0.75f, 0.75f));
        // Floor, ceiling, back and front walls.
        builder.AddQuad(white, Vec3{2.0f, 0.0f, 2.0f}, Vec3{2.0f, 0.0f, 0.0f},
                        Vec3{0.0f, 0.0f, 2.0f}, Vec3{0.0f, 1.0f, 0.0f});
        builder.AddQuad(white, Vec3{2.0f, 3.0f, 2.0f}, Vec3{2.0f, 0.0f, 0.0f},
                        Vec3{0.0f, 0.0f, 2.0f}, Vec3{0.0f, -1.0f, 0.0f});
        builder.AddQuad(white, Vec3{2.0f, 1.5f, 0.0f}, Vec3{2.0f, 0.0f, 0.0f},
                        Vec3{0.0f, 1.5f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f});
        builder.AddQuad(white, Vec3{2.0f, 1.5f, 4.0f}, Vec3{2.0f, 0.0f, 0.0f},
                        Vec3{0.0f, 1.5f, 0.0f}, Vec3{0.0f, 0.0f, -1.0f});
        builder.AddQuad(red, Vec3{0.0f, 1.5f, 2.0f}, Vec3{0.0f, 1.5f, 0.0f}, Vec3{0.0f, 0.0f, 2.0f},
                        Vec3{1.0f, 0.0f, 0.0f});
        builder.AddQuad(green, Vec3{4.0f, 1.5f, 2.0f}, Vec3{0.0f, 1.5f, 0.0f},
                        Vec3{0.0f, 0.0f, 2.0f}, Vec3{-1.0f, 0.0f, 0.0f});
        builder.AddBox(block, Vec3{1.0f, 0.0f, 1.6f}, Vec3{2.0f, 1.2f, 2.6f}, false);
        if (!builder.Build())
        {
            return false;
        }
        lamp.position = Vec3{2.0f, 2.7f, 2.0f};
        lamp.intensity = Vec3{12.0f, 12.0f, 12.0f};
        lamp.range = 20.0f;
        scene = builder.Scene();
        scene.pointLights = Span<const GiPointLight>(&lamp, 1);
        return true;
    }

    GiTestScene builder;
    GiPointLight lamp{};
    GiScene scene{};
};

// Points on the floor, the two coloured walls, and the block's top, each with
// the surface normal. Points inside the block are skipped.
std::vector<SamplePoint> CornellSamples()
{
    std::vector<SamplePoint> points;
    for (u32 ix = 0; ix < 7; ++ix)
    {
        for (u32 iz = 0; iz < 7; ++iz)
        {
            const f32 x = 0.5f + 0.5f * static_cast<f32>(ix);
            const f32 z = 0.5f + 0.5f * static_cast<f32>(iz);
            if (x > 0.95f && x < 2.05f && z > 1.55f && z < 2.65f)
            {
                continue;
            }
            points.push_back({Vec3{x, 0.0f, z}, Vec3{0.0f, 1.0f, 0.0f}});
        }
    }
    for (u32 iy = 0; iy < 3; ++iy)
    {
        for (u32 iz = 0; iz < 3; ++iz)
        {
            const f32 y = 0.6f + 0.9f * static_cast<f32>(iy);
            const f32 z = 0.8f + 1.2f * static_cast<f32>(iz);
            points.push_back({Vec3{0.0f, y, z}, Vec3{1.0f, 0.0f, 0.0f}});
            points.push_back({Vec3{4.0f, y, z}, Vec3{-1.0f, 0.0f, 0.0f}});
        }
    }
    points.push_back({Vec3{1.5f, 1.2f, 2.1f}, Vec3{0.0f, 1.0f, 0.0f}});
    return points;
}

std::vector<Vec3> GroundTruth(const GiScene& scene, const std::vector<SamplePoint>& points)
{
    PathTraceSettings settings{};
    settings.maxBounces = 6;
    std::vector<Vec3> reference;
    reference.reserve(points.size());
    for (u32 index = 0; index < points.size(); ++index)
    {
        reference.push_back(EstimateIrradiance(scene, points[index].position, points[index].normal,
                                               settings, 2048, 1000u + index));
    }
    return reference;
}

std::vector<Vec3> SampleVolume(const IrradianceVolume& volume,
                               const std::vector<SamplePoint>& points,
                               const IrradianceVolumeSampleDesc& desc = {})
{
    std::vector<Vec3> samples;
    samples.reserve(points.size());
    for (const SamplePoint& point : points)
    {
        samples.push_back(SampleIrradianceVolume(volume, point.position, point.normal, desc));
    }
    return samples;
}

Span<const Vec3> AsSpan(const std::vector<Vec3>& values)
{
    return Span<const Vec3>(values.data(), values.size());
}

}  // namespace

CG_TEST(software_gi_acceptance_volume_matches_ground_truth)
{
    HeapAllocator allocator;
    Jobs jobs(allocator);
    CornellRoom room(allocator);
    CG_REQUIRE(room.Build());

    const Vec3 volumeMin{0.25f, 0.25f, 0.25f};
    const Vec3 volumeMax{3.75f, 2.75f, 3.75f};
    IrradianceVolume volume = MakeVolume(allocator, volumeMin, volumeMax, 8, 6, 8);
    CG_REQUIRE(volume.IsValid());
    IrradianceBakeDesc bake{};
    bake.raysPerProbe = 512;
    bake.path.maxBounces = 4;
    IrradianceBakeStats stats{};
    CG_REQUIRE(BakeIrradianceVolume(room.scene, bake, volume, &stats, &jobs.system).IsOk());
    // The block buries some probes; they are found and excluded.
    CG_EXPECT_TRUE(stats.insideProbes > 0);
    CG_EXPECT_EQ(stats.validProbes + stats.insideProbes, volume.ProbeCount());

    const std::vector<SamplePoint> points = CornellSamples();
    const std::vector<Vec3> reference = GroundTruth(room.scene, points);
    const std::vector<Vec3> baked = SampleVolume(volume, points);

    IrradianceErrorMetrics metrics{};
    CG_REQUIRE(
        MeasureIrradianceError(AsSpan(baked), AsSpan(reference), 0.05f, 0.35f, metrics).IsOk());
    std::printf("[gi-acceptance] volume vs ground truth: %u samples, mean rel %.3f, max rel %.3f, "
                "bias %+.3f, energy %.3f, failing %.1f%%  (%u valid, %u inside probes)\n",
                metrics.sampleCount, metrics.meanRelativeError, metrics.maxRelativeError,
                metrics.meanRelativeBias, metrics.energyRatio, 100.0f * metrics.failingFraction,
                stats.validProbes, stats.insideProbes);
    CG_EXPECT_TRUE(metrics.meanRelativeError < 0.12f);
    CG_EXPECT_TRUE(metrics.energyRatio > 0.92f && metrics.energyRatio < 1.08f);
    CG_EXPECT_TRUE(metrics.failingFraction < 0.1f);

    // Control: a bake without interreflection. The probes see only surfaces
    // lit directly by the lamp, so the volume misses every further bounce and
    // the same thresholds fail.
    IrradianceVolume flat = MakeVolume(allocator, volumeMin, volumeMax, 8, 6, 8);
    IrradianceBakeDesc noBounce = bake;
    noBounce.path.maxBounces = 0;
    CG_REQUIRE(BakeIrradianceVolume(room.scene, noBounce, flat, nullptr, &jobs.system).IsOk());
    IrradianceErrorMetrics control{};
    CG_REQUIRE(MeasureIrradianceError(AsSpan(SampleVolume(flat, points)), AsSpan(reference), 0.05f,
                                      0.35f, control)
                   .IsOk());
    std::printf("[gi-acceptance] control (no bounces): mean rel %.3f, bias %+.3f, energy %.3f\n",
                control.meanRelativeError, control.meanRelativeBias, control.energyRatio);
    CG_EXPECT_FALSE(control.meanRelativeError < 0.12f &&
                    (control.energyRatio > 0.92f && control.energyRatio < 1.08f));
    CG_EXPECT_TRUE(control.energyRatio < 0.92f);

    // Colour bleeding, the visible signature of interreflection: the floor
    // beside the red wall is red-tinted and beside the green wall green-tinted,
    // in the bake as in the ground truth.
    const Vec3 nearRed =
        SampleIrradianceVolume(volume, Vec3{0.3f, 0.0f, 3.2f}, Vec3{0.0f, 1.0f, 0.0f});
    const Vec3 nearGreen =
        SampleIrradianceVolume(volume, Vec3{3.7f, 0.0f, 3.2f}, Vec3{0.0f, 1.0f, 0.0f});
    std::printf("[gi-acceptance] colour bleed: near red r/g %.2f, near green g/r %.2f\n",
                nearRed.x / nearRed.y, nearGreen.y / nearGreen.x);
    CG_EXPECT_TRUE(nearRed.x > 1.3f * nearRed.y);
    CG_EXPECT_TRUE(nearGreen.y > 1.3f * nearGreen.x);
}

// Two rooms sharing a wall. The lamp is in room A; room B is sealed, so its
// ground truth is black. The lattice is placed so that probes sit 0.35 either
// side of the wall, which is the configuration in which an unweighted
// interpolation pulls room A's light straight through.
CG_TEST(software_gi_acceptance_no_light_leak_through_walls)
{
    HeapAllocator allocator;
    Jobs jobs(allocator);
    GiTestScene builder(allocator);
    const u32 shell = builder.AddGroup(Diffuse(0.75f, 0.75f, 0.75f));
    const u32 wall = builder.AddGroup(Diffuse(0.75f, 0.75f, 0.75f));
    builder.AddBox(shell, Vec3{0.0f, 0.0f, 0.0f}, Vec3{8.0f, 3.0f, 4.0f}, true);
    builder.AddBox(wall, Vec3{3.95f, 0.0f, 0.0f}, Vec3{4.05f, 3.0f, 4.0f}, false);
    CG_REQUIRE(builder.Build());
    GiScene scene = builder.Scene();
    GiPointLight lamp{};
    lamp.position = Vec3{2.0f, 2.5f, 2.0f};
    lamp.intensity = Vec3{12.0f, 12.0f, 12.0f};
    lamp.range = 20.0f;
    scene.pointLights = Span<const GiPointLight>(&lamp, 1);

    IrradianceVolume volume =
        MakeVolume(allocator, Vec3{0.4f, 0.4f, 0.4f}, Vec3{7.6f, 2.6f, 3.6f}, 10, 4, 5);
    CG_REQUIRE(volume.IsValid());
    IrradianceBakeDesc bake{};
    bake.raysPerProbe = 256;
    bake.path.maxBounces = 3;
    CG_REQUIRE(BakeIrradianceVolume(scene, bake, volume, nullptr, &jobs.system).IsOk());

    // Room B surfaces within one probe spacing of the wall: floor, and the
    // wall's own face.
    std::vector<SamplePoint> dark;
    for (f32 x : {4.1f, 4.2f, 4.3f})
    {
        for (f32 z : {1.0f, 2.0f, 3.0f})
        {
            dark.push_back({Vec3{x, 0.0f, z}, Vec3{0.0f, 1.0f, 0.0f}});
        }
    }
    for (f32 y : {0.5f, 1.5f, 2.5f})
    {
        for (f32 z : {1.0f, 2.0f, 3.0f})
        {
            dark.push_back({Vec3{4.05f, y, z}, Vec3{1.0f, 0.0f, 0.0f}});
        }
    }

    // Ground truth: room B really is dark, and room A's floor by the wall is
    // what a leak would be borrowing from.
    PathTraceSettings settings{};
    settings.maxBounces = 4;
    const Vec3 truthDark = EstimateIrradiance(scene, Vec3{4.2f, 0.0f, 2.0f}, Vec3{0.0f, 1.0f, 0.0f},
                                              settings, 512, 7u);
    const Vec3 lit = EstimateIrradiance(scene, Vec3{3.8f, 0.0f, 2.0f}, Vec3{0.0f, 1.0f, 0.0f},
                                        settings, 2048, 8u);
    CG_EXPECT_EQ(truthDark.x, 0.0f);
    CG_REQUIRE(lit.x > 0.0f);

    // A dark-side sample counts as leaking above 2% of the lit side's
    // irradiance.
    constexpr f32 kLeakThreshold = 0.02f;
    LightLeakMetrics guarded{};
    CG_REQUIRE(
        MeasureLightLeak(AsSpan(SampleVolume(volume, dark)), lit, kLeakThreshold, guarded).IsOk());
    IrradianceVolumeSampleDesc unguarded{};
    unguarded.useVisibility = false;
    unguarded.useNormalWeighting = false;
    LightLeakMetrics leaking{};
    CG_REQUIRE(MeasureLightLeak(AsSpan(SampleVolume(volume, dark, unguarded)), lit, kLeakThreshold,
                                leaking)
                   .IsOk());
    std::printf("[gi-acceptance] leak through wall: guarded mean %.4f max %.4f (%u/%u over 2%%); "
                "unguarded mean %.4f max %.4f (%u/%u)\n",
                guarded.meanLeakRatio, guarded.maxLeakRatio, guarded.leakingSamples,
                guarded.sampleCount, leaking.meanLeakRatio, leaking.maxLeakRatio,
                leaking.leakingSamples, leaking.sampleCount);
    CG_EXPECT_TRUE(guarded.maxLeakRatio < kLeakThreshold);
    CG_EXPECT_EQ(guarded.leakingSamples, u32{0});
    CG_EXPECT_TRUE(leaking.maxLeakRatio > kLeakThreshold);
    CG_EXPECT_TRUE(leaking.leakingSamples > leaking.sampleCount / 2);

    // Room A next to the wall is still lit by the guarded lookup: leak
    // suppression must not turn into darkening the lit side.
    const Vec3 litBaked =
        SampleIrradianceVolume(volume, Vec3{3.8f, 0.0f, 2.0f}, Vec3{0.0f, 1.0f, 0.0f});
    std::printf("[gi-acceptance] lit side by the wall: baked %.3f vs truth %.3f\n", litBaked.x,
                lit.x);
    CG_EXPECT_TRUE(litBaked.x > 0.75f * lit.x);
}

// A ray-traced soft shadow at one sample per pixel, denoised. An orthographic
// camera looks straight down at a floor; a square blocker hovers above it and a
// disk light above that, so the shadow has an umbra and a wide penumbra.
CG_TEST(software_gi_acceptance_denoised_ray_traced_shadow)
{
    HeapAllocator allocator;
    GiTestScene builder(allocator);
    const u32 floor = builder.AddGroup(Diffuse(0.5f, 0.5f, 0.5f));
    const u32 blocker = builder.AddGroup(Diffuse(0.5f, 0.5f, 0.5f));
    builder.AddQuad(floor, Vec3{}, Vec3{10.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 10.0f},
                    Vec3{0.0f, 1.0f, 0.0f});
    builder.AddQuad(blocker, Vec3{0.0f, 1.0f, 0.0f}, Vec3{0.6f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 0.6f},
                    Vec3{0.0f, 1.0f, 0.0f});
    CG_REQUIRE(builder.Build());
    const TopLevelAccelerationStructure& geometry = builder.Geometry();

    constexpr i32 kSize = 96;
    constexpr f32 kExtent = 2.5f;
    const Vec3 lightCenter{0.4f, 4.0f, 0.2f};
    constexpr f32 kLightRadius = 0.9f;
    const auto floorPoint = [](i32 x, i32 y)
    {
        return Vec3{
            (static_cast<f32>(x) + 0.5f) / static_cast<f32>(kSize) * 2.0f * kExtent - kExtent, 0.0f,
            (static_cast<f32>(y) + 0.5f) / static_cast<f32>(kSize) * 2.0f * kExtent - kExtent};
    };
    // Visibility of one point on the disk light, chosen by two uniforms.
    const auto visibility = [&](const Vec3& point, f32 u1, f32 u2)
    {
        const f32 radius = kLightRadius * std::sqrt(u1);
        const f32 angle = 2.0f * kPi * u2;
        const Vec3 sample =
            lightCenter + Vec3{radius * std::cos(angle), 0.0f, radius * std::sin(angle)};
        return TraceVisibility(geometry, point, sample) ? 1.0f : 0.0f;
    };

    Framebuffer reference = MakeImage(allocator, kSize, kSize);
    Framebuffer noisy = MakeImage(allocator, kSize, kSize);
    Framebuffer output = MakeImage(allocator, kSize, kSize);
    Framebuffer normals = MakeImage(allocator, kSize, kSize);
    DepthBuffer depth = MakeDepth(allocator, kSize, kSize);
    CG_REQUIRE(reference.IsValid() && noisy.IsValid() && output.IsValid() && normals.IsValid() &&
               depth.IsValid());
    normals.Clear(EncodeNormal(Vec3{0.0f, 1.0f, 0.0f}));
    for (i32 y = 0; y < kSize; ++y)
    {
        f32* row = depth.Row(y);
        for (i32 x = 0; x < kSize; ++x)
        {
            row[x] = ReverseZFromDistance(5.0f, 0.1f, 100.0f);
            // Converged reference: 16 x 16 stratified samples of the disk.
            f32 sum = 0.0f;
            for (u32 sy = 0; sy < 16; ++sy)
            {
                for (u32 sx = 0; sx < 16; ++sx)
                {
                    sum += visibility(floorPoint(x, y), (static_cast<f32>(sx) + 0.5f) / 16.0f,
                                      (static_cast<f32>(sy) + 0.5f) / 16.0f);
                }
            }
            const f32 value = sum / 256.0f;
            reference.SetPixel(x, y, Color{value, value, value, 1.0f});
        }
    }

    const auto renderNoisy = [&](u32 frame)
    {
        for (i32 y = 0; y < kSize; ++y)
        {
            for (i32 x = 0; x < kSize; ++x)
            {
                const f32 value = visibility(
                    floorPoint(x, y),
                    HashUniform(static_cast<u32>(x), static_cast<u32>(y), frame * 2u),
                    HashUniform(static_cast<u32>(x), static_cast<u32>(y), frame * 2u + 1u));
                noisy.SetPixel(x, y, Color{value, value, value, 1.0f});
            }
        }
    };

    DenoiseGuides guides{};
    guides.depth = &depth;
    guides.normals = &normals;
    guides.nearPlane = 0.1f;
    guides.farPlane = 100.0f;
    auto created = SpatioTemporalDenoiser::Create(kSize, kSize, allocator);
    CG_REQUIRE(created.IsOk());
    SpatioTemporalDenoiser denoiser = std::move(created).Value();

    renderNoisy(0);
    NoiseMetrics raw{};
    CG_REQUIRE(MeasureNoise(noisy, reference, raw).IsOk());
    CG_REQUIRE(denoiser.Denoise(noisy, guides, DenoiserDesc{}, output).IsOk());
    NoiseMetrics single{};
    CG_REQUIRE(MeasureNoise(output, reference, single).IsOk());

    NoiseMetrics accumulated{};
    for (u32 frame = 1; frame < 8; ++frame)
    {
        renderNoisy(frame);
        CG_REQUIRE(denoiser.Denoise(noisy, guides, DenoiserDesc{}, output).IsOk());
    }
    CG_REQUIRE(MeasureNoise(output, reference, accumulated).IsOk());

    std::printf("[gi-acceptance] ray-traced shadow, 1 spp: raw rel RMSE %.3f bias %+.4f laplacian "
                "%.3f; denoised 1 frame %.3f %+.4f %.3f; 8 frames %.3f %+.4f %.3f\n",
                raw.relativeRmse, raw.relativeBias, raw.laplacianEnergy, single.relativeRmse,
                single.relativeBias, single.laplacianEnergy, accumulated.relativeRmse,
                accumulated.relativeBias, accumulated.laplacianEnergy);
    // Thresholds: one frame within 6% RMSE of the converged result, eight
    // within 3.5%, bias within 2% either way. The raw one-sample signal is the
    // control and fails the single-frame bar by a wide margin.
    CG_EXPECT_TRUE(single.relativeRmse < 0.06f);
    CG_EXPECT_TRUE(accumulated.relativeRmse < 0.035f);
    CG_EXPECT_TRUE(std::fabs(single.relativeBias) < 0.02f);
    CG_EXPECT_TRUE(std::fabs(accumulated.relativeBias) < 0.02f);
    CG_EXPECT_FALSE(raw.relativeRmse < 0.06f);
    CG_EXPECT_TRUE(raw.relativeRmse > 2.0f * single.relativeRmse);
    CG_EXPECT_TRUE(accumulated.relativeRmse < single.relativeRmse);
}
