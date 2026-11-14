// Phase 28 visual-acceptance measurements.
//
// The roadmap's Phase 28 bar is a test scene with no shadow acne, no detached
// shadows, and no visible seam between cascades. These tests render a real
// four-cascade directional shadow set over a ground plane and a long wall,
// then measure the four artifacts through the same sampler the PBR pixel
// shader uses:
//
//   * acne           - unoccluded receivers must stay fully lit, including the
//                      wall's own light-facing face, which is the surface that
//                      self-shadows when the depth bias is wrong;
//   * detachment     - the wall's shadow must start at its contact point and
//                      stay unbroken across its analytic extent, and must not
//                      extend past it;
//   * cascade seams  - visibility along a transect that crosses every cascade
//                      split at a constant distance from the shadow edge must
//                      be continuous, because the ideal value is constant
//                      there and any step is cascade disagreement;
//   * shimmer        - a static receiver's visibility must not flicker while
//                      the camera translates and the cascades re-snap.
//
// Each test also drives a deliberately mis-biased configuration through the
// same metric, so a passing threshold proves the measurement has teeth rather
// than that it cannot fail.

#include <cyber/jobs/job_system.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/software/pbr.h>
#include <cyber/software/shadow_quality.h>
#include <cyber/test/test.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>

using namespace cyber::memory;
using namespace cyber::software;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::math::Color;
using cyber::math::Vec3;

namespace
{

// Scene constants. The wall runs along Z so its ground shadow is a straight
// band of constant width at every camera depth; that is what makes a
// constant-offset transect a valid seam measurement.
constexpr f32 kWallHalfThickness = 0.15f;
constexpr f32 kWallHeight = 2.0f;
constexpr f32 kGroundExtent = 80.0f;
constexpr f32 kShadowDistance = 40.0f;
constexpr i32 kMapResolution = 512;
constexpr u32 kCascadeCount = 4;

// Light travels toward +X and -Y, so the wall's shadow falls toward +X.
const Vec3 kLightTravel = cyber::math::Normalize(Vec3{1.0f, -1.2f, 0.0f});
const Vec3 kDirectionToLight = cyber::math::Normalize(Vec3{-1.0f, 1.2f, 0.0f});

// Where the wall's shadow ends on the ground: the top edge projects along the
// light by height * (L.x / -L.y) from the wall's far side.
constexpr f32 kShadowFarEdge = kWallHalfThickness + kWallHeight * (1.0f / 1.2f);

constexpr u32 kSceneVertexCount = 6 + 36;

void AppendQuad(VertexInput* vertices, u32& count, const Vec3& a, const Vec3& b, const Vec3& c,
                const Vec3& d, const Vec3& normal)
{
    const Vec3 order[6] = {a, b, c, a, c, d};
    for (const Vec3& position : order)
    {
        VertexInput& vertex = vertices[count++];
        vertex.position = position;
        vertex.normal = normal;
        vertex.color = Color{1.0f, 1.0f, 1.0f, 1.0f};
    }
}

void AppendBox(VertexInput* vertices, u32& count, const Vec3& center, const Vec3& halfExtents)
{
    const f32 x0 = center.x - halfExtents.x;
    const f32 x1 = center.x + halfExtents.x;
    const f32 y0 = center.y - halfExtents.y;
    const f32 y1 = center.y + halfExtents.y;
    const f32 z0 = center.z - halfExtents.z;
    const f32 z1 = center.z + halfExtents.z;

    AppendQuad(vertices, count, Vec3{x0, y0, z1}, Vec3{x1, y0, z1}, Vec3{x1, y1, z1},
               Vec3{x0, y1, z1}, Vec3{0.0f, 0.0f, 1.0f});
    AppendQuad(vertices, count, Vec3{x1, y0, z0}, Vec3{x0, y0, z0}, Vec3{x0, y1, z0},
               Vec3{x1, y1, z0}, Vec3{0.0f, 0.0f, -1.0f});
    AppendQuad(vertices, count, Vec3{x0, y0, z0}, Vec3{x0, y0, z1}, Vec3{x0, y1, z1},
               Vec3{x0, y1, z0}, Vec3{-1.0f, 0.0f, 0.0f});
    AppendQuad(vertices, count, Vec3{x1, y0, z1}, Vec3{x1, y0, z0}, Vec3{x1, y1, z0},
               Vec3{x1, y1, z1}, Vec3{1.0f, 0.0f, 0.0f});
    AppendQuad(vertices, count, Vec3{x0, y1, z1}, Vec3{x1, y1, z1}, Vec3{x1, y1, z0},
               Vec3{x0, y1, z0}, Vec3{0.0f, 1.0f, 0.0f});
    AppendQuad(vertices, count, Vec3{x0, y0, z0}, Vec3{x1, y0, z0}, Vec3{x1, y0, z1},
               Vec3{x0, y0, z1}, Vec3{0.0f, -1.0f, 0.0f});
}

void BuildScene(VertexInput* vertices, u32& count)
{
    count = 0;
    AppendQuad(vertices, count, Vec3{-kGroundExtent, 0.0f, kGroundExtent},
               Vec3{kGroundExtent, 0.0f, kGroundExtent}, Vec3{kGroundExtent, 0.0f, -kGroundExtent},
               Vec3{-kGroundExtent, 0.0f, -kGroundExtent}, Vec3{0.0f, 1.0f, 0.0f});
    AppendBox(vertices, count, Vec3{0.0f, kWallHeight * 0.5f, -25.0f},
              Vec3{kWallHalfThickness, kWallHeight * 0.5f, 35.0f});
}

Camera MakeAcceptanceCamera()
{
    Camera camera{};
    // Looking along -Z with no pitch, so a ground probe's positive view depth
    // is exactly -z and a transect in z sweeps the cascade splits directly.
    camera.position = Vec3{0.0f, 2.0f, 0.0f};
    camera.yawRadians = 0.0f;
    camera.pitchRadians = 0.0f;
    camera.nearPlane = 0.1f;
    camera.farPlane = 60.0f;
    return camera;
}

DirectionalShadowCascadeConfig MakeAcceptanceConfig()
{
    DirectionalShadowCascadeConfig config{};
    config.cascadeCount = kCascadeCount;
    config.mapResolution = static_cast<u32>(kMapResolution);
    config.shadowDistance = kShadowDistance;
    config.splitLambda = 0.7f;
    config.blendFraction = 0.12f;
    config.depthPadding = 30.0f;
    config.filterRadius = 2;
    return config;
}

// Owns the four cascade depth maps for one acceptance scene and rebuilds the
// cascade set for a given camera. The depth maps outlive every sampler call
// made through `cascades`, which is what the borrowed-view contract requires.
struct CascadeScene
{
    DepthBuffer depth[kMaxDirectionalShadowCascades]{};
    DirectionalShadowCascadeSet cascades{};
    PipelineStats stats[kMaxDirectionalShadowCascades]{};

    bool Allocate(HeapAllocator& heap)
    {
        for (u32 index = 0; index < kCascadeCount; ++index)
        {
            auto created =
                DepthBuffer::Create(DepthBufferDesc{kMapResolution, kMapResolution, 0}, heap);
            if (!created)
            {
                return false;
            }
            depth[index] = std::move(created.Value());
        }
        return true;
    }

    // `jobs` may be null; the rasterizer writes identical bytes either way, so
    // the acceptance thresholds do not depend on the schedule.
    bool Render(HeapAllocator& heap, const Camera& camera,
                const DirectionalShadowCascadeConfig& config, const VertexInput* vertices,
                u32 vertexCount, cyber::jobs::JobSystem* jobs = nullptr)
    {
        cascades = DirectionalShadowCascadeSet{};
        if (!BuildDirectionalShadowCascades(camera, 16.0f / 9.0f, kLightTravel, config, cascades))
        {
            return false;
        }
        return static_cast<bool>(RenderDirectionalShadowMaps(
            cascades, cyber::memory::Span<const VertexInput>(vertices, vertexCount),
            cyber::memory::Span<DepthBuffer>(depth, kCascadeCount), heap, jobs, CullMode::None,
            jobs != nullptr, cyber::memory::Span<PipelineStats>(stats, kCascadeCount)));
    }
};

ShadowProbe GroundProbe(f32 x, f32 z)
{
    ShadowProbe probe{};
    probe.worldPosition = Vec3{x, 0.0f, z};
    probe.worldNormal = Vec3{0.0f, 1.0f, 0.0f};
    probe.directionToLight = kDirectionToLight;
    // The camera looks along -Z from z = 0, so view depth is -z.
    probe.viewDepth = -z;
    return probe;
}

// A probe on the wall's light-facing face, which is the surface most prone to
// self-shadowing because its own depth is what the map stores.
ShadowProbe WallProbe(f32 y, f32 z)
{
    ShadowProbe probe{};
    probe.worldPosition = Vec3{-kWallHalfThickness, y, z};
    probe.worldNormal = Vec3{-1.0f, 0.0f, 0.0f};
    probe.directionToLight = kDirectionToLight;
    probe.viewDepth = -z;
    return probe;
}

}  // namespace

CG_TEST(phase28_acceptance_scene_shows_no_shadow_acne)
{
    HeapAllocator heap;
    VertexInput vertices[kSceneVertexCount]{};
    u32 vertexCount = 0;
    BuildScene(vertices, vertexCount);
    CG_REQUIRE(vertexCount == kSceneVertexCount);

    CascadeScene scene;
    CG_REQUIRE(scene.Allocate(heap));
    const Camera camera = MakeAcceptanceCamera();
    const DirectionalShadowCascadeConfig config = MakeAcceptanceConfig();
    CG_REQUIRE(scene.Render(heap, camera, config, vertices, vertexCount));
    for (u32 index = 0; index < kCascadeCount; ++index)
    {
        CG_EXPECT_TRUE(scene.stats[index].pixels > 0);
    }

    // Unoccluded ground on both sides of the shadow band, sampled across the
    // whole shadow distance so every cascade is exercised.
    constexpr u32 kGroundColumns = 20;
    constexpr u32 kGroundRows = 30;
    constexpr u32 kWallRows = 16;
    constexpr u32 kWallColumns = 20;
    constexpr u32 kAcneProbeCount = 2 * kGroundColumns * kGroundRows + kWallRows * kWallColumns;
    ShadowProbe probes[kAcneProbeCount]{};
    u32 probeCount = 0;
    for (u32 row = 0; row < kGroundRows; ++row)
    {
        const f32 z = -1.5f - static_cast<f32>(row) * 1.25f;
        for (u32 column = 0; column < kGroundColumns; ++column)
        {
            const f32 offset = 0.6f + static_cast<f32>(column) * 0.95f;
            // Lit side (toward the light) and beyond the shadow's far edge.
            probes[probeCount++] = GroundProbe(-offset, z);
            probes[probeCount++] = GroundProbe(kShadowFarEdge + offset, z);
        }
    }
    // The wall's own light-facing face.
    for (u32 row = 0; row < kWallRows; ++row)
    {
        const f32 y = 0.1f + static_cast<f32>(row) * 0.115f;
        for (u32 column = 0; column < kWallColumns; ++column)
        {
            const f32 z = -1.5f - static_cast<f32>(column) * 1.9f;
            probes[probeCount++] = WallProbe(y, z);
        }
    }
    CG_REQUIRE(probeCount == kAcneProbeCount);

    ShadowAcneMetrics acne{};
    CG_REQUIRE(static_cast<bool>(MeasureShadowAcne(
        scene.cascades, cyber::memory::Span<const ShadowProbe>(probes, probeCount), 0.999f, acne)));
    std::printf("[phase28] acne: samples=%u darkened=%u fraction=%.6f min=%.6f mean=%.6f run=%u\n",
                acne.sampleCount, acne.darkenedSamples, static_cast<double>(acne.darkenedFraction),
                static_cast<double>(acne.minVisibility), static_cast<double>(acne.meanVisibility),
                acne.longestDarkenedRun);
    CG_EXPECT_EQ(acne.sampleCount, probeCount);
    CG_EXPECT_EQ(acne.darkenedSamples, u32{0});
    CG_EXPECT_NEAR(acne.minVisibility, 1.0f, 1.0e-6f);
    CG_EXPECT_NEAR(acne.meanVisibility, 1.0f, 1.0e-6f);

    // Removing the bias entirely is the classic acne configuration: a lit
    // receiver's own quantized depth loses its own comparison. The same probe
    // set must detect it, which is what makes the threshold above meaningful.
    DirectionalShadowCascadeConfig zeroBiasConfig = config;
    zeroBiasConfig.constantDepthBias = 0.0f;
    zeroBiasConfig.slopeDepthBias = 0.0f;
    zeroBiasConfig.normalOffsetBias = 0.0f;
    CascadeScene unbiased;
    CG_REQUIRE(unbiased.Allocate(heap));
    CG_REQUIRE(unbiased.Render(heap, camera, zeroBiasConfig, vertices, vertexCount));
    ShadowAcneMetrics zeroBiasAcne{};
    CG_REQUIRE(static_cast<bool>(MeasureShadowAcne(
        unbiased.cascades, cyber::memory::Span<const ShadowProbe>(probes, probeCount), 0.999f,
        zeroBiasAcne)));
    std::printf("[phase28] acne (zero bias): darkened=%u fraction=%.6f min=%.6f\n",
                zeroBiasAcne.darkenedSamples, static_cast<double>(zeroBiasAcne.darkenedFraction),
                static_cast<double>(zeroBiasAcne.minVisibility));
    CG_EXPECT_TRUE(zeroBiasAcne.darkenedSamples > 0);
    CG_EXPECT_TRUE(zeroBiasAcne.minVisibility < acne.minVisibility);

    // A negative reverse-Z bias moves the receiver away from the light and can
    // only cause acne, so the builder rejects it outright.
    DirectionalShadowCascadeConfig negativeConfig = config;
    negativeConfig.constantDepthBias = -0.002f;
    DirectionalShadowCascadeSet rejected{};
    CG_EXPECT_FALSE(static_cast<bool>(BuildDirectionalShadowCascades(
        camera, 16.0f / 9.0f, kLightTravel, negativeConfig, rejected)));
}

CG_TEST(phase28_acceptance_scene_shadow_stays_attached_to_its_caster)
{
    HeapAllocator heap;
    VertexInput vertices[kSceneVertexCount]{};
    u32 vertexCount = 0;
    BuildScene(vertices, vertexCount);

    CascadeScene scene;
    CG_REQUIRE(scene.Allocate(heap));
    const Camera camera = MakeAcceptanceCamera();
    const DirectionalShadowCascadeConfig config = MakeAcceptanceConfig();
    CG_REQUIRE(scene.Render(heap, camera, config, vertices, vertexCount));

    // Walk from directly beneath the wall outward along the shadow, stopping
    // short of the analytic far edge so every sample must be occluded.
    constexpr f32 kStep = 0.05f;
    constexpr u32 kWalkSamples = 28;
    static_assert(kStep * static_cast<f32>(kWalkSamples) < kShadowFarEdge,
                  "the contact walk must stay inside the analytic shadow extent");
    ShadowContactMetrics contact{};
    CG_REQUIRE(static_cast<bool>(MeasureShadowContact(scene.cascades, GroundProbe(0.0f, -6.0f),
                                                      Vec3{1.0f, 0.0f, 0.0f}, kStep, kWalkSamples,
                                                      0.5f, contact)));
    std::printf("[phase28] contact: visibility=%.6f lit=%d distance=%.4f firstLit=%u\n",
                static_cast<double>(contact.contactVisibility), contact.foundLitSample ? 1 : 0,
                static_cast<double>(contact.detachmentDistance), contact.firstLitIndex);
    CG_EXPECT_TRUE(contact.contactVisibility < 0.02f);
    CG_EXPECT_FALSE(contact.foundLitSample);
    CG_EXPECT_NEAR(contact.detachmentDistance, kStep * static_cast<f32>(kWalkSamples), 1.0e-5f);

    // The shadow must also stop where the geometry says it does, so the band
    // is neither detached nor smeared past its caster.
    ShadowAcneMetrics beyond{};
    ShadowProbe beyondProbes[16]{};
    for (u32 index = 0; index < 16; ++index)
    {
        beyondProbes[index] =
            GroundProbe(kShadowFarEdge + 0.6f, -2.0f - static_cast<f32>(index) * 2.2f);
    }
    CG_REQUIRE(static_cast<bool>(MeasureShadowAcne(
        scene.cascades, cyber::memory::Span<const ShadowProbe>(beyondProbes, 16), 0.999f, beyond)));
    CG_EXPECT_EQ(beyond.darkenedSamples, u32{0});

    // An exaggerated normal offset is what detaches a shadow in practice: the
    // receiver is pushed toward the light until its own contact point samples
    // lit. The same walk must report that gap.
    DirectionalShadowCascadeConfig detachedConfig = config;
    detachedConfig.normalOffsetBias = 1.4f;
    detachedConfig.constantDepthBias = 0.02f;
    CascadeScene detached;
    CG_REQUIRE(detached.Allocate(heap));
    CG_REQUIRE(detached.Render(heap, camera, detachedConfig, vertices, vertexCount));
    ShadowContactMetrics detachedContact{};
    CG_REQUIRE(static_cast<bool>(MeasureShadowContact(detached.cascades, GroundProbe(0.0f, -6.0f),
                                                      Vec3{1.0f, 0.0f, 0.0f}, kStep, kWalkSamples,
                                                      0.5f, detachedContact)));
    std::printf("[phase28] contact (detached bias): visibility=%.6f lit=%d distance=%.4f\n",
                static_cast<double>(detachedContact.contactVisibility),
                detachedContact.foundLitSample ? 1 : 0,
                static_cast<double>(detachedContact.detachmentDistance));
    CG_EXPECT_TRUE(detachedContact.foundLitSample);
    CG_EXPECT_TRUE(detachedContact.detachmentDistance < contact.detachmentDistance);
}

CG_TEST(phase28_acceptance_scene_crosses_cascade_splits_without_a_seam)
{
    HeapAllocator heap;
    VertexInput vertices[kSceneVertexCount]{};
    u32 vertexCount = 0;
    BuildScene(vertices, vertexCount);

    CascadeScene scene;
    CG_REQUIRE(scene.Allocate(heap));
    const Camera camera = MakeAcceptanceCamera();
    const DirectionalShadowCascadeConfig config = MakeAcceptanceConfig();
    CG_REQUIRE(scene.Render(heap, camera, config, vertices, vertexCount));
    CG_REQUIRE(scene.cascades.count == kCascadeCount);

    // The wall has constant height and the light is constant, so the ideal
    // visibility at a fixed distance from the shadow edge does not depend on
    // depth. Every step along these transects is cascade disagreement.
    //
    // The probes are spaced finely enough (0.05 world units of depth) that the
    // overlap blend's ramp is resolved as many small steps. A coarser transect
    // would report the ramp itself as a large step and could not tell a smooth
    // blend from a hard seam.
    constexpr u32 kTransectSamples = 720;
    constexpr f32 kTransectSpacing = 0.05f;
    // Measured worst case for this scene: the coarse far cascade's shadow edge
    // lands one texel short of the near cascade's, so this offset is shadowed
    // in the near cascade and lit in the far one. It is the hardest position
    // for the overlap blend to hide.
    constexpr f32 kSeamOffset = -0.02f;
    ShadowProbe umbra[kTransectSamples]{};
    ShadowProbe penumbra[kTransectSamples]{};
    for (u32 index = 0; index < kTransectSamples; ++index)
    {
        const f32 z = -2.0f - static_cast<f32>(index) * kTransectSpacing;
        umbra[index] = GroundProbe(kShadowFarEdge * 0.5f, z);
        penumbra[index] = GroundProbe(kShadowFarEdge + kSeamOffset, z);
    }

    ShadowTransectMetrics umbraMetrics{};
    CG_REQUIRE(static_cast<bool>(MeasureShadowTransect(
        scene.cascades, cyber::memory::Span<const ShadowProbe>(umbra, kTransectSamples),
        umbraMetrics)));
    std::printf("[phase28] umbra transect: maxStep=%.6f atSplit=%.6f mean=%.6f crossings=%u\n",
                static_cast<double>(umbraMetrics.maxStep),
                static_cast<double>(umbraMetrics.maxStepAcrossSplit),
                static_cast<double>(umbraMetrics.meanStep), umbraMetrics.splitCrossings);
    CG_EXPECT_EQ(umbraMetrics.splitCrossings, kCascadeCount - 1);
    CG_EXPECT_TRUE(umbraMetrics.maxStep < 0.01f);
    CG_EXPECT_TRUE(umbraMetrics.maxStepAcrossSplit < 0.01f);

    ShadowTransectMetrics penumbraMetrics{};
    CG_REQUIRE(static_cast<bool>(MeasureShadowTransect(
        scene.cascades, cyber::memory::Span<const ShadowProbe>(penumbra, kTransectSamples),
        penumbraMetrics)));
    std::printf("[phase28] penumbra transect: maxStep=%.6f atSplit=%.6f mean=%.6f\n",
                static_cast<double>(penumbraMetrics.maxStep),
                static_cast<double>(penumbraMetrics.maxStepAcrossSplit),
                static_cast<double>(penumbraMetrics.meanStep));
    CG_EXPECT_EQ(penumbraMetrics.splitCrossings, kCascadeCount - 1);
    // Adjacent cascades quantize the same edge to different texel grids, so
    // they disagree by up to a full comparison at this offset. The overlap
    // blend must spread that disagreement into small increments instead of a
    // step, both at the split itself and anywhere along the ramp.
    CG_EXPECT_TRUE(penumbraMetrics.maxStepAcrossSplit < 0.1f);
    CG_EXPECT_TRUE(penumbraMetrics.maxStep < 0.15f);

    // Disabling the overlap blend is what makes a split visible, so the same
    // transect must measure a larger step without it.
    DirectionalShadowCascadeConfig hardConfig = config;
    hardConfig.blendFraction = 0.0f;
    CascadeScene hard;
    CG_REQUIRE(hard.Allocate(heap));
    CG_REQUIRE(hard.Render(heap, camera, hardConfig, vertices, vertexCount));
    ShadowTransectMetrics hardMetrics{};
    CG_REQUIRE(static_cast<bool>(MeasureShadowTransect(
        hard.cascades, cyber::memory::Span<const ShadowProbe>(penumbra, kTransectSamples),
        hardMetrics)));
    std::printf("[phase28] penumbra transect (no blend): maxStep=%.6f atSplit=%.6f\n",
                static_cast<double>(hardMetrics.maxStep),
                static_cast<double>(hardMetrics.maxStepAcrossSplit));
    // Without the blend the same transect steps by a full comparison, which is
    // the seam the blend exists to remove.
    CG_EXPECT_TRUE(hardMetrics.maxStepAcrossSplit > 0.9f);
    CG_EXPECT_TRUE(hardMetrics.maxStepAcrossSplit > penumbraMetrics.maxStepAcrossSplit * 4.0f);
}

CG_TEST(phase28_acceptance_scene_keeps_shadows_stable_while_the_camera_moves)
{
    HeapAllocator heap;
    VertexInput vertices[kSceneVertexCount]{};
    u32 vertexCount = 0;
    BuildScene(vertices, vertexCount);

    CascadeScene scene;
    CG_REQUIRE(scene.Allocate(heap));
    const DirectionalShadowCascadeConfig config = MakeAcceptanceConfig();

    // Static world probes: deep umbra, fully lit ground, and the wall's own
    // light-facing face. None of them may change while the camera moves.
    constexpr u32 kProbeCount = 24;
    ShadowProbe probes[kProbeCount]{};
    for (u32 index = 0; index < 8; ++index)
    {
        const f32 z = -3.0f - static_cast<f32>(index) * 4.3f;
        probes[index] = GroundProbe(kShadowFarEdge * 0.5f, z);
        probes[index + 8] = GroundProbe(-4.0f - static_cast<f32>(index), z);
        probes[index + 16] = WallProbe(0.4f + static_cast<f32>(index) * 0.15f, z);
    }

    // The camera translates perpendicular to the light, which is the motion
    // that makes an unsnapped cascade crawl. Steps are a mix of sub-texel and
    // multi-texel so both the snap and the re-render are exercised.
    constexpr u32 kFrameCount = 24;
    constexpr f32 kFrameStep = 0.028f;
    auto jobsResult = cyber::jobs::JobSystem::Create(cyber::jobs::JobSystem::Desc{}, heap);
    CG_REQUIRE(static_cast<bool>(jobsResult));
    cyber::jobs::JobSystem jobs = std::move(jobsResult.Value());

    f32 sequence[kProbeCount * kFrameCount]{};
    for (u32 frame = 0; frame < kFrameCount; ++frame)
    {
        Camera camera = MakeAcceptanceCamera();
        camera.position.x += kFrameStep * static_cast<f32>(frame);
        camera.position.z -= kFrameStep * 0.5f * static_cast<f32>(frame);
        CG_REQUIRE(scene.Render(heap, camera, config, vertices, vertexCount, &jobs));
        for (u32 probe = 0; probe < kProbeCount; ++probe)
        {
            ShadowProbe moved = probes[probe];
            // View depth follows the camera, exactly as the shader computes it.
            moved.viewDepth = -(moved.worldPosition.z - camera.position.z);
            sequence[static_cast<cyber::core::usize>(frame) * kProbeCount + probe] =
                EvaluateDirectionalShadowVisibility(scene.cascades, moved.viewDepth,
                                                    moved.worldPosition, moved.worldNormal,
                                                    moved.directionToLight);
        }
    }

    ShadowStabilityMetrics stability{};
    CG_REQUIRE(static_cast<bool>(
        MeasureShadowStability(cyber::memory::Span<const f32>(sequence, kProbeCount * kFrameCount),
                               kProbeCount, kFrameCount, stability)));
    std::printf(
        "[phase28] stability: maxDelta=%.6f (probe=%u frame=%u) mean=%.6f maxVariation=%.6f\n",
        static_cast<double>(stability.maxFrameDelta), stability.maxFrameDeltaProbe,
        stability.maxFrameDeltaFrame, static_cast<double>(stability.meanFrameDelta),
        static_cast<double>(stability.maxTotalVariation));
    CG_EXPECT_EQ(stability.probeCount, kProbeCount);
    CG_EXPECT_EQ(stability.frameCount, kFrameCount);
    CG_EXPECT_TRUE(stability.maxFrameDelta < 0.01f);
    CG_EXPECT_TRUE(stability.maxTotalVariation < 0.05f);

    // Shadow edges are where crawl shows first, because the penumbra straddles
    // texels the snap can only align to a whole texel. Sampled on the edge,
    // visibility is allowed to move, but texel snapping must keep each frame's
    // change small instead of letting the edge jump between frames.
    constexpr u32 kEdgeProbeCount = 12;
    ShadowProbe edges[kEdgeProbeCount]{};
    for (u32 index = 0; index < kEdgeProbeCount; ++index)
    {
        edges[index] = GroundProbe(kShadowFarEdge, -3.0f - static_cast<f32>(index) * 2.9f);
    }
    f32 edgeSequence[kEdgeProbeCount * kFrameCount]{};
    for (u32 frame = 0; frame < kFrameCount; ++frame)
    {
        Camera camera = MakeAcceptanceCamera();
        camera.position.x += kFrameStep * static_cast<f32>(frame);
        camera.position.z -= kFrameStep * 0.5f * static_cast<f32>(frame);
        CG_REQUIRE(scene.Render(heap, camera, config, vertices, vertexCount, &jobs));
        for (u32 probe = 0; probe < kEdgeProbeCount; ++probe)
        {
            const f32 viewDepth = -(edges[probe].worldPosition.z - camera.position.z);
            edgeSequence[static_cast<cyber::core::usize>(frame) * kEdgeProbeCount + probe] =
                EvaluateDirectionalShadowVisibility(
                    scene.cascades, viewDepth, edges[probe].worldPosition, edges[probe].worldNormal,
                    edges[probe].directionToLight);
        }
    }
    ShadowStabilityMetrics edgeStability{};
    CG_REQUIRE(static_cast<bool>(MeasureShadowStability(
        cyber::memory::Span<const f32>(edgeSequence, kEdgeProbeCount * kFrameCount),
        kEdgeProbeCount, kFrameCount, edgeStability)));
    std::printf("[phase28] edge stability: maxDelta=%.6f mean=%.6f maxVariation=%.6f\n",
                static_cast<double>(edgeStability.maxFrameDelta),
                static_cast<double>(edgeStability.meanFrameDelta),
                static_cast<double>(edgeStability.maxTotalVariation));
    CG_EXPECT_TRUE(edgeStability.maxFrameDelta < 0.05f);
    CG_EXPECT_TRUE(edgeStability.maxTotalVariation < 0.1f);
}

CG_TEST(phase28_shadow_quality_metrics_reject_invalid_input)
{
    DirectionalShadowCascadeSet cascades{};
    ShadowAcneMetrics acne{};
    CG_EXPECT_FALSE(static_cast<bool>(
        MeasureShadowAcne(cascades, cyber::memory::Span<const ShadowProbe>{}, 0.5f, acne)));

    const ShadowProbe probe = GroundProbe(0.0f, -1.0f);
    CG_EXPECT_FALSE(static_cast<bool>(MeasureShadowAcne(
        cascades, cyber::memory::Span<const ShadowProbe>(&probe, 1), 0.0f, acne)));
    CG_EXPECT_FALSE(static_cast<bool>(MeasureShadowAcne(
        cascades, cyber::memory::Span<const ShadowProbe>(&probe, 1), 1.5f, acne)));

    ShadowProbe infinite = probe;
    infinite.worldPosition.y = std::numeric_limits<f32>::infinity();
    CG_EXPECT_FALSE(static_cast<bool>(MeasureShadowAcne(
        cascades, cyber::memory::Span<const ShadowProbe>(&infinite, 1), 0.5f, acne)));

    ShadowContactMetrics contact{};
    CG_EXPECT_FALSE(static_cast<bool>(
        MeasureShadowContact(cascades, probe, Vec3{1.0f, 0.0f, 0.0f}, 0.1f, 0, 0.5f, contact)));
    CG_EXPECT_FALSE(static_cast<bool>(
        MeasureShadowContact(cascades, probe, Vec3{1.0f, 0.0f, 0.0f}, 0.0f, 4, 0.5f, contact)));
    CG_EXPECT_FALSE(static_cast<bool>(
        MeasureShadowContact(cascades, probe, Vec3{0.0f, 0.0f, 0.0f}, 0.1f, 4, 0.5f, contact)));

    ShadowTransectMetrics transect{};
    CG_EXPECT_FALSE(static_cast<bool>(MeasureShadowTransect(
        cascades, cyber::memory::Span<const ShadowProbe>(&probe, 1), transect)));
    const ShadowProbe unordered[2] = {GroundProbe(0.0f, -4.0f), GroundProbe(0.0f, -1.0f)};
    CG_EXPECT_FALSE(static_cast<bool>(MeasureShadowTransect(
        cascades, cyber::memory::Span<const ShadowProbe>(unordered, 2), transect)));

    // An ordered transect over an empty cascade set is lit everywhere, so it
    // is continuous and reports no split crossing.
    const ShadowProbe ordered[2] = {GroundProbe(0.0f, -1.0f), GroundProbe(0.0f, -4.0f)};
    CG_REQUIRE(static_cast<bool>(MeasureShadowTransect(
        cascades, cyber::memory::Span<const ShadowProbe>(ordered, 2), transect)));
    CG_EXPECT_NEAR(transect.maxStep, 0.0f, 1.0e-6f);
    CG_EXPECT_EQ(transect.splitCrossings, u32{0});

    ShadowStabilityMetrics stability{};
    const f32 values[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    CG_EXPECT_FALSE(static_cast<bool>(
        MeasureShadowStability(cyber::memory::Span<const f32>(values, 4), 0, 2, stability)));
    CG_EXPECT_FALSE(static_cast<bool>(
        MeasureShadowStability(cyber::memory::Span<const f32>(values, 4), 3, 2, stability)));
    CG_REQUIRE(static_cast<bool>(
        MeasureShadowStability(cyber::memory::Span<const f32>(values, 4), 4, 1, stability)));
    CG_EXPECT_NEAR(stability.maxFrameDelta, 0.0f, 1.0e-6f);

    const f32 flicker[4] = {0.0f, 1.0f, 0.0f, 1.0f};
    CG_REQUIRE(static_cast<bool>(
        MeasureShadowStability(cyber::memory::Span<const f32>(flicker, 4), 1, 4, stability)));
    CG_EXPECT_NEAR(stability.maxFrameDelta, 1.0f, 1.0e-6f);
    CG_EXPECT_NEAR(stability.maxTotalVariation, 3.0f, 1.0e-6f);
}
