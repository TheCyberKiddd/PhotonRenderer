#include "gi_test_scenes.h"

#include <cyber/jobs/job_system.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/software/global_illumination.h>
#include <cyber/software/pbr.h>
#include <cyber/test/test.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

using namespace cyber::software;
using cyber::core::f32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::math::Vec3;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using cyber::software::test::ConstantRadianceSh;
using cyber::software::test::Diffuse;
using cyber::software::test::Emissive;
using cyber::software::test::GiTestScene;

namespace
{

constexpr f32 kPi = cyber::math::kPi;

f32 Luminance(const Vec3& value)
{
    return 0.2126f * value.x + 0.7152f * value.y + 0.0722f * value.z;
}

PathTraceSettings Exact(u32 bounces)
{
    PathTraceSettings settings{};
    settings.maxBounces = bounces;
    settings.russianRoulette = false;
    return settings;
}

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
    if (created.IsError())
    {
        return IrradianceVolume{};
    }
    return std::move(created).Value();
}

RayDesc Ray(const Vec3& origin, const Vec3& direction)
{
    RayDesc ray{};
    ray.origin = origin;
    ray.direction = direction;
    return ray;
}

}  // namespace

// The white furnace. Inside a closed box whose every wall emits radiance Le
// and reflects a fraction rho, the radiance after n bounces is exactly
// Le * (1 + rho + ... + rho^n) in every direction, because every ray hits a
// wall. With roulette off the estimator has no variance at all, so energy
// conservation is checked to rounding, not to a statistical tolerance.
CG_TEST(software_gi_white_furnace_is_exact)
{
    HeapAllocator allocator;
    GiTestScene builder(allocator);
    const u32 walls = builder.AddGroup(Emissive(0.5f, 1.0f));
    builder.AddBox(walls, Vec3{-1.0f, -1.0f, -1.0f}, Vec3{1.0f, 1.0f, 1.0f}, true);
    CG_REQUIRE(builder.Build());
    const GiScene scene = builder.Scene();
    CG_REQUIRE(ValidateGiScene(scene).IsOk());

    const f32 expected[4] = {1.0f, 1.5f, 1.75f, 1.875f};
    for (u32 bounces = 0; bounces < 4; ++bounces)
    {
        const Vec3 irradiance = EstimateIrradiance(scene, Vec3{0.1f, -0.2f, 0.3f},
                                                   Vec3{0.3f, 1.0f, -0.2f}, Exact(bounces), 64, 1u);
        CG_EXPECT_NEAR(irradiance.x, kPi * expected[bounces], 1.0e-3f);
        CG_EXPECT_NEAR(irradiance.y, kPi * expected[bounces], 1.0e-3f);
    }

    // Unbiased with roulette: the infinite series sums to Le / (1 - rho) = 2.
    PathTraceSettings roulette{};
    roulette.maxBounces = 32;
    roulette.russianRoulette = true;
    roulette.rouletteStartBounce = 1;
    PathTraceStats stats{};
    const Vec3 converged = EstimateIrradiance(scene, Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f},
                                              roulette, 8192, 9u, &stats);
    CG_EXPECT_NEAR(converged.x / kPi, 2.0f, 0.06f);
    CG_EXPECT_TRUE(stats.terminatedByRoulette > 0);
    CG_EXPECT_TRUE(stats.rays > 8192);
}

// A huge lit floor seen from just above: every hemisphere ray lands on the
// floor, whose outgoing radiance is rho / pi times the light's irradiance, so
// the downward irradiance is exactly rho times it.
CG_TEST(software_gi_directional_light_and_shadow)
{
    HeapAllocator allocator;
    GiTestScene builder(allocator);
    const u32 floor = builder.AddGroup(Diffuse(0.5f, 0.5f, 0.5f));
    const u32 blocker = builder.AddGroup(Diffuse(0.5f, 0.5f, 0.5f));
    builder.AddQuad(floor, Vec3{}, Vec3{1000.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 1000.0f},
                    Vec3{0.0f, 1.0f, 0.0f});
    builder.AddQuad(blocker, Vec3{0.0f, 1.0f, 0.0f}, Vec3{0.5f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 0.5f},
                    Vec3{0.0f, 1.0f, 0.0f});
    CG_REQUIRE(builder.Build());
    GiScene scene = builder.Scene();
    GiDirectionalLight sun{};
    sun.direction = Vec3{0.0f, -1.0f, 0.0f};
    sun.irradiance = Vec3{2.0f, 2.0f, 2.0f};
    scene.directionalLights = Span<const GiDirectionalLight>(&sun, 1);

    // Lit floor away from the blocker: radiance rho / pi * E.
    const Vec3 lit =
        TracePathRadiance(scene, Ray({4.0f, 0.5f, 0.0f}, {0.0f, -1.0f, 0.0f}), Exact(0), 0u, 0u);
    CG_EXPECT_NEAR(lit.x, 0.5f * 2.0f / kPi, 1.0e-5f);
    // The floor under the blocker receives no direct light, and with no
    // bounces nothing else.
    const Vec3 shadowed =
        TracePathRadiance(scene, Ray({0.0f, 0.5f, 0.0f}, {0.0f, -1.0f, 0.0f}), Exact(0), 0u, 0u);
    CG_EXPECT_NEAR(shadowed.x, 0.0f, 1.0e-6f);
    // With a bounce the shadowed floor picks up light reflected by the
    // blocker's underside, which itself sees the lit floor: interreflection.
    const Vec3 bounced =
        TracePathRadiance(scene, Ray({0.0f, 0.5f, 0.0f}, {0.0f, -1.0f, 0.0f}), Exact(2), 0u, 0u);
    CG_EXPECT_TRUE(bounced.x >= 0.0f);

    // A downward-facing point just above the open floor.
    const Vec3 downward = EstimateIrradiance(scene, Vec3{20.0f, 0.01f, 0.0f},
                                             Vec3{0.0f, -1.0f, 0.0f}, Exact(0), 256, 3u);
    CG_EXPECT_NEAR(downward.x, 0.5f * 2.0f, 2.0e-3f);
}

CG_TEST(software_gi_point_light_and_sky)
{
    HeapAllocator allocator;
    GiTestScene builder(allocator);
    const u32 floor = builder.AddGroup(Diffuse(0.5f, 0.5f, 0.5f));
    builder.AddQuad(floor, Vec3{}, Vec3{1000.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 1000.0f},
                    Vec3{0.0f, 1.0f, 0.0f});
    CG_REQUIRE(builder.Build());
    GiScene scene = builder.Scene();

    GiPointLight lamp{};
    lamp.position = Vec3{0.0f, 2.0f, 0.0f};
    lamp.intensity = Vec3{8.0f, 8.0f, 8.0f};
    lamp.range = 100.0f;
    scene.pointLights = Span<const GiPointLight>(&lamp, 1);
    const Vec3 below =
        TracePathRadiance(scene, Ray({0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}), Exact(0), 0u, 0u);
    const f32 window = (1.0f - 0.02f * 0.02f) * (1.0f - 0.02f * 0.02f);
    // The shadow ray starts a ray offset off the floor, which shortens the
    // distance to the lamp by about 1e-4; the tolerance allows for exactly
    // that and nothing more.
    const f32 expectedBelow = 0.5f / kPi * 8.0f / 4.0f * window;
    CG_EXPECT_NEAR(below.x, expectedBelow, expectedBelow * 3.0e-4f);

    // Out of range: nothing.
    lamp.range = 1.5f;
    const Vec3 outOfRange =
        TracePathRadiance(scene, Ray({0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}), Exact(0), 0u, 0u);
    CG_EXPECT_NEAR(outOfRange.x, 0.0f, 1.0e-7f);

    // A constant sky seen by an upward-facing point on the floor: every ray
    // escapes, so the irradiance is pi times the sky radiance exactly.
    scene.pointLights = Span<const GiPointLight>();
    scene.skyRadiance = Vec3{0.3f, 0.2f, 0.1f};
    const Vec3 sky = EstimateIrradiance(scene, Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f},
                                        Exact(4), 128, 5u);
    CG_EXPECT_NEAR(sky.x, kPi * 0.3f, 1.0e-4f);
    CG_EXPECT_NEAR(sky.z, kPi * 0.1f, 1.0e-4f);
}

CG_TEST(software_gi_scene_validation)
{
    HeapAllocator allocator;
    GiScene empty{};
    CG_EXPECT_TRUE(ValidateGiScene(empty).IsError());
    // Held in a named value: CG_EXPECT_EQ binds a reference to its argument,
    // and a member of a returned temporary would not outlive the statement.
    const Vec3 emptyIrradiance =
        EstimateIrradiance(empty, Vec3{}, Vec3{0.0f, 1.0f, 0.0f}, Exact(1), 16, 0u);
    CG_EXPECT_EQ(emptyIrradiance.x, 0.0f);

    GiTestScene builder(allocator);
    const u32 floor = builder.AddGroup(Diffuse(0.5f, 0.5f, 0.5f));
    builder.AddQuad(floor, Vec3{}, Vec3{1.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f},
                    Vec3{0.0f, 1.0f, 0.0f});
    CG_REQUIRE(builder.Build());
    GiScene scene = builder.Scene();
    CG_EXPECT_TRUE(ValidateGiScene(scene).IsOk());

    GiDirectionalLight lights[kMaxGiDirectionalLights + 1]{};
    scene.directionalLights = Span<const GiDirectionalLight>(lights, kMaxGiDirectionalLights + 1);
    CG_EXPECT_TRUE(ValidateGiScene(scene).IsError());
    scene.directionalLights = Span<const GiDirectionalLight>(lights, 1);
    lights[0].irradiance.x = std::numeric_limits<f32>::infinity();
    CG_EXPECT_TRUE(ValidateGiScene(scene).IsError());
    lights[0].irradiance.x = 1.0f;
    lights[0].direction = Vec3{};
    CG_EXPECT_TRUE(ValidateGiScene(scene).IsError());
    lights[0].direction = Vec3{0.0f, -1.0f, 0.0f};
    CG_EXPECT_TRUE(ValidateGiScene(scene).IsOk());
    scene.rayOffsetScale = -1.0f;
    CG_EXPECT_TRUE(ValidateGiScene(scene).IsError());
    scene.rayOffsetScale = 1.0f;

    // An identifier with no material entry uses the default rather than
    // reading past the table.
    scene.materials = Span<const GiSurfaceMaterial>();
    const Vec3 lit =
        TracePathRadiance(scene, Ray({0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}), Exact(0), 0u, 0u);
    CG_EXPECT_NEAR(lit.x, GiSurfaceMaterial{}.albedo.x / kPi, 1.0e-5f);
}

CG_TEST(software_gi_estimates_are_deterministic)
{
    HeapAllocator allocator;
    GiTestScene builder(allocator);
    const u32 room = builder.AddGroup(Diffuse(0.7f, 0.7f, 0.7f));
    builder.AddBox(room, Vec3{-2.0f, 0.0f, -2.0f}, Vec3{2.0f, 3.0f, 2.0f}, true);
    CG_REQUIRE(builder.Build());
    GiScene scene = builder.Scene();
    GiPointLight lamp{};
    lamp.position = Vec3{0.0f, 2.5f, 0.0f};
    lamp.intensity = Vec3{10.0f, 10.0f, 10.0f};
    scene.pointLights = Span<const GiPointLight>(&lamp, 1);

    PathTraceSettings settings{};
    const Vec3 a = EstimateIrradiance(scene, Vec3{1.0f, 0.0f, 1.0f}, Vec3{0.0f, 1.0f, 0.0f},
                                      settings, 200, 77u);
    const Vec3 b = EstimateIrradiance(scene, Vec3{1.0f, 0.0f, 1.0f}, Vec3{0.0f, 1.0f, 0.0f},
                                      settings, 200, 77u);
    CG_EXPECT_EQ(std::memcmp(&a, &b, sizeof(Vec3)), 0);
    const Vec3 c = EstimateIrradiance(scene, Vec3{1.0f, 0.0f, 1.0f}, Vec3{0.0f, 1.0f, 0.0f},
                                      settings, 200, 78u);
    CG_EXPECT_NE(std::memcmp(&a, &c, sizeof(Vec3)), 0);
    CG_EXPECT_TRUE(Luminance(a) > 0.0f);
}

CG_TEST(software_gi_octahedral_mapping_round_trips)
{
    // Axes, diagonals, both hemispheres, and the fold.
    const Vec3 directions[] = {{1.0f, 0.0f, 0.0f},  {-1.0f, 0.0f, 0.0f},  {0.0f, 1.0f, 0.0f},
                               {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f},   {0.0f, 0.0f, -1.0f},
                               {0.3f, -0.4f, 0.8f}, {-0.6f, 0.2f, -0.7f}, {0.5f, 0.5f, -0.1f}};
    for (const Vec3& raw : directions)
    {
        const Vec3 direction = cyber::math::Normalize(raw);
        const cyber::math::Vec2 uv = EncodeOctahedral(direction);
        CG_EXPECT_TRUE(uv.x >= 0.0f && uv.x <= 1.0f && uv.y >= 0.0f && uv.y <= 1.0f);
        const Vec3 decoded = DecodeOctahedral(uv);
        CG_EXPECT_NEAR(decoded.x, direction.x, 1.0e-5f);
        CG_EXPECT_NEAR(decoded.y, direction.y, 1.0e-5f);
        CG_EXPECT_NEAR(decoded.z, direction.z, 1.0e-5f);
    }
    // A degenerate direction maps to the centre rather than NaN.
    const cyber::math::Vec2 centre = EncodeOctahedral(Vec3{});
    CG_EXPECT_NEAR(centre.x, 0.5f, 1.0e-6f);
}

CG_TEST(software_gi_volume_layout_and_validation)
{
    HeapAllocator allocator;
    IrradianceVolumeDesc desc{};
    desc.boundsMin = Vec3{0.0f, 0.0f, 0.0f};
    desc.boundsMax = Vec3{0.0f, 1.0f, 1.0f};
    CG_EXPECT_TRUE(IrradianceVolume::Create(desc, allocator).IsError());
    desc.boundsMax = Vec3{2.0f, 1.0f, 4.0f};
    desc.probeCountX = 1;
    CG_EXPECT_TRUE(IrradianceVolume::Create(desc, allocator).IsError());
    desc.probeCountX = kMaxIrradianceProbesPerAxis + 1;
    CG_EXPECT_TRUE(IrradianceVolume::Create(desc, allocator).IsError());
    desc.probeCountX = 128;
    desc.probeCountY = 128;
    desc.probeCountZ = 128;
    CG_EXPECT_TRUE(IrradianceVolume::Create(desc, allocator).IsError());

    IrradianceVolume volume =
        MakeVolume(allocator, Vec3{0.0f, 0.0f, 0.0f}, Vec3{2.0f, 1.0f, 4.0f}, 3, 2, 5);
    CG_REQUIRE(volume.IsValid());
    CG_EXPECT_EQ(volume.ProbeCount(), u32{30});
    CG_EXPECT_NEAR(volume.ProbeSpacing().x, 1.0f, 1.0e-6f);
    CG_EXPECT_NEAR(volume.ProbeSpacing().y, 1.0f, 1.0e-6f);
    CG_EXPECT_NEAR(volume.ProbeSpacing().z, 1.0f, 1.0e-6f);
    CG_EXPECT_EQ(volume.ProbeIndex(2, 1, 4), u32{29});
    CG_EXPECT_EQ(volume.ProbeIndex(3, 0, 0), kInvalidRayIndex);
    CG_EXPECT_NEAR(volume.ProbePosition(29).z, 4.0f, 1.0e-6f);
    CG_EXPECT_NEAR(volume.ProbePosition(volume.ProbeIndex(1, 1, 2)).x, 1.0f, 1.0e-6f);
    CG_EXPECT_TRUE(volume.DataSizeBytes() >= sizeof(IrradianceProbe) * 30);
    CG_EXPECT_EQ(volume.Probe(1000).state, IrradianceProbeState::Unbaked);

    IrradianceProbe probe{};
    probe.state = IrradianceProbeState::Valid;
    volume.SetProbe(3, probe);
    volume.SetProbe(1000, probe);
    CG_EXPECT_EQ(volume.Probe(3).state, IrradianceProbeState::Valid);
    volume.Clear();
    CG_EXPECT_EQ(volume.Probe(3).state, IrradianceProbeState::Unbaked);

    // An unbaked volume answers with the fallback.
    IrradianceVolumeSampleDesc sampling{};
    sampling.fallbackIrradiance = Vec3{0.25f, 0.5f, 0.75f};
    const IrradianceVolumeSample sample = SampleIrradianceVolumeDetailed(
        volume, Vec3{1.0f, 0.5f, 2.0f}, Vec3{0.0f, 1.0f, 0.0f}, sampling);
    CG_EXPECT_TRUE(sample.usedFallback);
    CG_EXPECT_EQ(sample.contributingProbes, u32{0});
    CG_EXPECT_NEAR(sample.irradiance.z, 0.75f, 1.0e-6f);

    IrradianceVolumeView view{};
    view.sampling = sampling;
    view.intensity = 2.0f;
    CG_EXPECT_NEAR(EvaluateIrradianceVolume(view, Vec3{}, Vec3{0.0f, 1.0f, 0.0f}).y, 1.0f, 1.0e-6f);
}

// A bake inside the furnace: the radiance field is constant, so every probe
// projects it exactly onto the DC term and every lookup, anywhere, for any
// normal, returns the same irradiance.
CG_TEST(software_gi_bake_reproduces_constant_field)
{
    HeapAllocator allocator;
    GiTestScene builder(allocator);
    const u32 walls = builder.AddGroup(Emissive(0.5f, 1.0f));
    builder.AddBox(walls, Vec3{-1.0f, -1.0f, -1.0f}, Vec3{1.0f, 1.0f, 1.0f}, true);
    CG_REQUIRE(builder.Build());
    const GiScene scene = builder.Scene();

    IrradianceVolume volume =
        MakeVolume(allocator, Vec3{-0.6f, -0.6f, -0.6f}, Vec3{0.6f, 0.6f, 0.6f}, 3, 3, 3);
    CG_REQUIRE(volume.IsValid());
    IrradianceBakeDesc bake{};
    bake.raysPerProbe = 512;
    bake.path = Exact(2);
    IrradianceBakeStats stats{};
    CG_REQUIRE(BakeIrradianceVolume(scene, bake, volume, &stats).IsOk());
    CG_EXPECT_EQ(stats.probeCount, u32{27});
    CG_EXPECT_EQ(stats.validProbes, u32{27});
    CG_EXPECT_EQ(stats.insideProbes, u32{0});
    CG_EXPECT_TRUE(stats.path.rays >= u64{27} * 512u * 3u);

    const f32 expected = kPi * 1.75f;
    const Vec3 normals[4] = {
        {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.5f, -0.5f, 0.7f}};
    const Vec3 points[3] = {{0.0f, 0.0f, 0.0f}, {0.3f, -0.2f, 0.5f}, {-0.55f, 0.55f, 0.1f}};
    for (const Vec3& point : points)
    {
        for (const Vec3& normal : normals)
        {
            const Vec3 irradiance = SampleIrradianceVolume(volume, point, normal);
            CG_EXPECT_NEAR(irradiance.x, expected, expected * 0.01f);
        }
    }
    // Distances are clamped to 1.5 times the probe spacing, 0.9 here. Every
    // wall is at least 1.0 from the centre probe, so every lobe's moments sit
    // exactly on the clamp.
    const IrradianceProbe& center = volume.Probe(volume.ProbeIndex(1, 1, 1));
    for (u32 texel = 0; texel < kIrradianceVisibilityTexels; ++texel)
    {
        CG_EXPECT_NEAR(center.meanDistance[texel], 0.9f, 1.0e-4f);
        CG_EXPECT_NEAR(center.meanDistanceSquared[texel], 0.81f, 1.0e-4f);
    }
    // The corner probe at (0.6, 0.6, 0.6) has the +X wall 0.4 away and the -X
    // wall 1.6 away: the map tells the two directions apart.
    const IrradianceProbe& corner = volume.Probe(volume.ProbeIndex(2, 2, 2));
    const auto texelToward = [](const Vec3& direction)
    {
        const cyber::math::Vec2 uv = EncodeOctahedral(direction);
        const u32 u = std::min(static_cast<u32>(uv.x * kIrradianceVisibilityResolution),
                               kIrradianceVisibilityResolution - 1);
        const u32 v = std::min(static_cast<u32>(uv.y * kIrradianceVisibilityResolution),
                               kIrradianceVisibilityResolution - 1);
        return v * kIrradianceVisibilityResolution + u;
    };
    const f32 towardNear = corner.meanDistance[texelToward(Vec3{1.0f, 0.0f, 0.0f})];
    const f32 towardFar = corner.meanDistance[texelToward(Vec3{-1.0f, 0.0f, 0.0f})];
    CG_EXPECT_TRUE(towardNear < 0.6f);
    // Rays tilted 30 degrees off -X reach the +Y and +Z walls just inside the
    // clamp, so the far texel reads a little under it, never near the 0.4 wall.
    CG_EXPECT_TRUE(towardFar > 0.8f);
    CG_EXPECT_TRUE(towardFar > towardNear + 0.3f);
    CG_EXPECT_NEAR(center.backfaceFraction, 0.0f, 1.0e-6f);
}

// A probe buried in a solid box sees nothing but back faces and is classified
// Inside; probes in the open around it stay Valid.
CG_TEST(software_gi_bake_classifies_buried_probes)
{
    HeapAllocator allocator;
    GiTestScene builder(allocator);
    const u32 solid = builder.AddGroup(Diffuse(0.5f, 0.5f, 0.5f));
    builder.AddBox(solid, Vec3{-0.5f, -0.5f, -0.5f}, Vec3{0.5f, 0.5f, 0.5f}, false);
    CG_REQUIRE(builder.Build());
    GiScene scene = builder.Scene();
    scene.skyRadiance = Vec3{1.0f, 1.0f, 1.0f};

    IrradianceVolume volume =
        MakeVolume(allocator, Vec3{-1.0f, -1.0f, -1.0f}, Vec3{1.0f, 1.0f, 1.0f}, 3, 3, 3);
    IrradianceBakeDesc bake{};
    bake.raysPerProbe = 128;
    bake.path = Exact(1);
    IrradianceBakeStats stats{};
    CG_REQUIRE(BakeIrradianceVolume(scene, bake, volume, &stats).IsOk());
    CG_EXPECT_EQ(stats.insideProbes, u32{1});
    CG_EXPECT_EQ(stats.validProbes, u32{26});
    const IrradianceProbe& buried = volume.Probe(volume.ProbeIndex(1, 1, 1));
    CG_EXPECT_EQ(buried.state, IrradianceProbeState::Inside);
    CG_EXPECT_NEAR(buried.backfaceFraction, 1.0f, 1.0e-6f);
    CG_EXPECT_EQ(Luminance(buried.radiance.coefficients[0]), 0.0f);

    // A lookup next to the box's top face does not use the buried probe: it
    // is lit by the sky, not darkened by the inside of the box.
    const IrradianceVolumeSample sample =
        SampleIrradianceVolumeDetailed(volume, Vec3{0.0f, 0.5f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f});
    CG_EXPECT_FALSE(sample.usedFallback);
    CG_EXPECT_TRUE(sample.contributingProbes <= 7);
    CG_EXPECT_TRUE(sample.irradiance.x > 0.5f * kPi);

    // Far outside the volume the nearest cell answers rather than the
    // fallback.
    const IrradianceVolumeSample outside =
        SampleIrradianceVolumeDetailed(volume, Vec3{50.0f, 50.0f, 50.0f}, Vec3{0.0f, 1.0f, 0.0f});
    CG_EXPECT_FALSE(outside.usedFallback);
}

CG_TEST(software_gi_bake_validation_leaves_volume_untouched)
{
    HeapAllocator allocator;
    GiTestScene builder(allocator);
    const u32 floor = builder.AddGroup(Diffuse(0.5f, 0.5f, 0.5f));
    builder.AddQuad(floor, Vec3{}, Vec3{5.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 5.0f},
                    Vec3{0.0f, 1.0f, 0.0f});
    CG_REQUIRE(builder.Build());
    const GiScene scene = builder.Scene();
    IrradianceVolume volume =
        MakeVolume(allocator, Vec3{-1.0f, 0.5f, -1.0f}, Vec3{1.0f, 1.5f, 1.0f}, 2, 2, 2);

    IrradianceBakeDesc bake{};
    bake.raysPerProbe = 8;
    CG_EXPECT_TRUE(BakeIrradianceVolume(scene, bake, volume).IsError());
    bake.raysPerProbe = 64;
    bake.insideBackfaceThreshold = 1.5f;
    CG_EXPECT_TRUE(BakeIrradianceVolume(scene, bake, volume).IsError());
    bake.insideBackfaceThreshold = 0.25f;
    bake.path.maxBounces = 33;
    CG_EXPECT_TRUE(BakeIrradianceVolume(scene, bake, volume).IsError());
    bake.path.maxBounces = 1;
    CG_EXPECT_TRUE(BakeIrradianceVolume(GiScene{}, bake, volume).IsError());
    IrradianceVolume invalid;
    CG_EXPECT_TRUE(BakeIrradianceVolume(scene, bake, invalid).IsError());
    for (u32 index = 0; index < volume.ProbeCount(); ++index)
    {
        CG_EXPECT_EQ(volume.Probe(index).state, IrradianceProbeState::Unbaked);
    }
    CG_EXPECT_TRUE(BakeIrradianceVolume(scene, bake, volume).IsOk());
    CG_EXPECT_EQ(volume.Probe(0).state, IrradianceProbeState::Valid);
}

// The bake is a pure function of its inputs: job-parallel and single-threaded
// runs produce identical bytes.
CG_TEST(software_gi_bake_is_deterministic_across_jobs)
{
    HeapAllocator allocator;
    GiTestScene builder(allocator);
    const u32 room = builder.AddGroup(Diffuse(0.7f, 0.7f, 0.7f));
    const u32 red = builder.AddGroup(Diffuse(0.8f, 0.1f, 0.1f));
    builder.AddBox(room, Vec3{-2.0f, 0.0f, -2.0f}, Vec3{2.0f, 3.0f, 2.0f}, true);
    builder.AddBox(red, Vec3{-0.5f, 0.0f, -0.5f}, Vec3{0.5f, 1.0f, 0.5f}, false);
    CG_REQUIRE(builder.Build());
    GiScene scene = builder.Scene();
    GiPointLight lamp{};
    lamp.position = Vec3{0.0f, 2.5f, 0.0f};
    lamp.intensity = Vec3{10.0f, 10.0f, 10.0f};
    scene.pointLights = Span<const GiPointLight>(&lamp, 1);

    IrradianceBakeDesc bake{};
    bake.raysPerProbe = 64;
    IrradianceVolume serial =
        MakeVolume(allocator, Vec3{-1.8f, 0.2f, -1.8f}, Vec3{1.8f, 2.8f, 1.8f}, 4, 3, 4);
    IrradianceVolume parallel =
        MakeVolume(allocator, Vec3{-1.8f, 0.2f, -1.8f}, Vec3{1.8f, 2.8f, 1.8f}, 4, 3, 4);
    bake.useJobs = false;
    CG_REQUIRE(BakeIrradianceVolume(scene, bake, serial).IsOk());

    cyber::jobs::JobSystem::Desc jobDesc{};
    jobDesc.workerCount = 4;
    auto jobs = cyber::jobs::JobSystem::Create(jobDesc, allocator);
    CG_REQUIRE(jobs.IsOk());
    bake.useJobs = true;
    CG_REQUIRE(BakeIrradianceVolume(scene, bake, parallel, nullptr, &jobs.Value()).IsOk());
    jobs.Value().Shutdown();

    // Bit-exact, member by member. A whole-struct memcmp also compared the
    // padding after the u8 state, whose bytes are whatever the baking thread's
    // stack held, so the test passed or failed by accident of thread layout.
    for (u32 index = 0; index < serial.ProbeCount(); ++index)
    {
        const IrradianceProbe& a = serial.Probe(index);
        const IrradianceProbe& b = parallel.Probe(index);
        CG_EXPECT_EQ(std::memcmp(a.radiance.coefficients, b.radiance.coefficients,
                                 sizeof(a.radiance.coefficients)),
                     0);
        CG_EXPECT_EQ(std::memcmp(a.meanDistance, b.meanDistance, sizeof(a.meanDistance)), 0);
        CG_EXPECT_EQ(std::memcmp(a.meanDistanceSquared, b.meanDistanceSquared,
                                 sizeof(a.meanDistanceSquared)),
                     0);
        CG_EXPECT_EQ(std::memcmp(&a.backfaceFraction, &b.backfaceFraction, sizeof(f32)), 0);
        CG_EXPECT_TRUE(a.state == b.state);
    }
    // The red box is a solid in the open; a probe is classified Inside only if
    // it falls within the box, and no probe of this lattice does.
    IrradianceBakeStats stats{};
    CG_REQUIRE(BakeIrradianceVolume(scene, bake, serial, &stats).IsOk());
    CG_EXPECT_EQ(stats.insideProbes, u32{0});
}

// The forward path consumes a volume exactly as it consumes a flat ambient
// irradiance of the same value, and inside the image-based path the volume
// replaces the environment's diffuse lookup.
CG_TEST(software_gi_pbr_integration)
{
    HeapAllocator allocator;
    IrradianceVolume volume =
        MakeVolume(allocator, Vec3{-1.0f, -1.0f, -1.0f}, Vec3{1.0f, 1.0f, 1.0f}, 2, 2, 2);
    CG_REQUIRE(volume.IsValid());
    IrradianceProbe probe{};
    probe.state = IrradianceProbeState::Valid;
    probe.radiance = ConstantRadianceSh(Vec3{0.4f, 0.3f, 0.2f});
    for (f32& distance : probe.meanDistance)
    {
        distance = 10.0f;
    }
    for (f32& distance : probe.meanDistanceSquared)
    {
        distance = 100.0f;
    }
    for (u32 index = 0; index < volume.ProbeCount(); ++index)
    {
        volume.SetProbe(index, probe);
    }
    const Vec3 normal{0.0f, 1.0f, 0.0f};
    const Vec3 irradiance = SampleIrradianceVolume(volume, Vec3{0.2f, 0.1f, -0.3f}, normal);
    CG_EXPECT_NEAR(irradiance.x, kPi * 0.4f, 1.0e-3f);
    CG_EXPECT_NEAR(irradiance.z, kPi * 0.2f, 1.0e-3f);

    Varyings varyings{};
    varyings.normal = normal;
    varyings.worldPosition = Vec3{0.2f, 0.1f, -0.3f};

    PbrLightingData flat{};
    flat.eyePosition = Vec3{0.0f, 3.0f, 1.0f};
    flat.ambientIrradiance = irradiance;
    flat.material.metallic = 0.0f;
    const cyber::math::Color expectedFlat = PbrPixelShader(varyings, &flat);

    IrradianceVolumeView view{};
    view.volume = &volume;
    PbrLightingData baked = flat;
    baked.ambientIrradiance = Vec3{100.0f, 100.0f, 100.0f};
    baked.irradianceVolume = &view;
    const cyber::math::Color actualBaked = PbrPixelShader(varyings, &baked);
    CG_EXPECT_NEAR(actualBaked.r, expectedFlat.r, 1.0e-5f);
    CG_EXPECT_NEAR(actualBaked.g, expectedFlat.g, 1.0e-5f);
    CG_EXPECT_NEAR(actualBaked.b, expectedFlat.b, 1.0e-5f);

    // Ambient occlusion darkens the baked term, which is what supplies the
    // contact shadowing the probe grid cannot.
    baked.ambientOcclusion = 0.5f;
    const cyber::math::Color occluded = PbrPixelShader(varyings, &baked);
    CG_EXPECT_NEAR(occluded.r, actualBaked.r * 0.5f, 1.0e-5f);

    // Image-based path: an environment whose diffuse SH equals the volume's
    // answer, against an environment with black diffuse plus the volume.
    const SphericalHarmonicsL2 matching = ConstantRadianceSh(Vec3{0.4f, 0.3f, 0.2f});
    const SphericalHarmonicsL2 black{};
    EnvironmentLighting withSh{};
    withSh.diffuseSh = &matching;
    EnvironmentLighting withoutSh{};
    withoutSh.diffuseSh = &black;
    PbrLightingData environment = flat;
    environment.environment = &withSh;
    const cyber::math::Color expectedIbl = PbrPixelShader(varyings, &environment);
    PbrLightingData environmentBaked = flat;
    environmentBaked.environment = &withoutSh;
    environmentBaked.irradianceVolume = &view;
    const cyber::math::Color actualIbl = PbrPixelShader(varyings, &environmentBaked);
    CG_EXPECT_NEAR(actualIbl.r, expectedIbl.r, 1.0e-4f);
    CG_EXPECT_NEAR(actualIbl.g, expectedIbl.g, 1.0e-4f);
    CG_EXPECT_NEAR(actualIbl.b, expectedIbl.b, 1.0e-4f);
    CG_EXPECT_TRUE(actualIbl.r > 0.0f);
}
