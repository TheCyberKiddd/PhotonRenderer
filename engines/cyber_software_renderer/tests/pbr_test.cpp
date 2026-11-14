#include <cyber/memory/heap_allocator.h>
#include <cyber/software/pbr.h>
#include <cyber/test/test.h>

#include <cmath>
#include <limits>
#include <utility>

using namespace cyber::software;
using namespace cyber::memory;
using cyber::core::f32;
using cyber::core::u32;
using cyber::math::Color;
using cyber::math::Vec3;

namespace
{

constexpr f32 kPi = cyber::math::kPi;

Varyings MakeSurface()
{
    Varyings surface{};
    surface.normal = Vec3{0.0f, 1.0f, 0.0f};
    surface.tangent = Vec3{1.0f, 0.0f, 0.0f};
    surface.worldPosition = Vec3::Zero();
    surface.color = Color{1.0f, 1.0f, 1.0f, 1.0f};
    return surface;
}

f32 IntegrateWhiteFurnace(const PbrMaterial& material)
{
    constexpr u32 kSamples = 32768;
    const Vec3 normal{0.0f, 1.0f, 0.0f};
    const Vec3 view{0.0f, 1.0f, 0.0f};
    f32 sum = 0.0f;
    for (u32 index = 0; index < kSamples; ++index)
    {
        const f32 y = (static_cast<f32>(index) + 0.5f) / static_cast<f32>(kSamples);
        const f32 phi = 2.0f * kPi * static_cast<f32>(index) * 0.61803398875f;
        const f32 radial = std::sqrt(1.0f - y * y);
        const Vec3 light{radial * std::cos(phi), y, radial * std::sin(phi)};
        const Color brdf = EvaluateGgxBrdf(normal, view, light, material);
        sum += brdf.r * y;
    }
    return sum * (2.0f * kPi / static_cast<f32>(kSamples));
}

}  // namespace

CG_TEST(software_pbr_ggx_brdf_is_finite_and_respects_hemisphere)
{
    PbrMaterial material{};
    material.baseColor = Color{0.8f, 0.4f, 0.2f, 1.0f};
    material.metallic = 0.65f;
    material.roughness = 0.3f;

    const Color front = EvaluateGgxBrdf(Vec3{0.0f, 1.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f},
                                        Vec3{0.0f, 1.0f, 0.0f}, material);
    CG_EXPECT_TRUE(std::isfinite(front.r));
    CG_EXPECT_TRUE(front.r > 0.0f);
    CG_EXPECT_TRUE(front.r > front.g);

    const Color back = EvaluateGgxBrdf(Vec3{0.0f, 1.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f},
                                       Vec3{0.0f, -1.0f, 0.0f}, material);
    CG_EXPECT_NEAR(back.r, 0.0f, 1.0e-7f);
    const Color invalid = EvaluateGgxBrdf(Vec3::Zero(), Vec3::Zero(), Vec3::Zero(), material);
    CG_EXPECT_NEAR(invalid.r, 0.0f, 1.0e-7f);
}

CG_TEST(software_pbr_ggx_terms_match_normal_incidence)
{
    const Vec3 normal{0.0f, 1.0f, 0.0f};
    const Color baseColor{0.8f, 0.4f, 0.2f, 1.0f};
    PbrMaterial material{};
    material.baseColor = baseColor;
    material.roughness = 1.0f;
    material.metallic = 1.0f;

    const Color metal = EvaluateGgxBrdf(normal, normal, normal, material);
    CG_EXPECT_NEAR(metal.r, baseColor.r / (4.0f * kPi), 1.0e-5f);
    CG_EXPECT_NEAR(metal.g, baseColor.g / (4.0f * kPi), 1.0e-5f);
    CG_EXPECT_NEAR(metal.b, baseColor.b / (4.0f * kPi), 1.0e-5f);

    material.metallic = 0.0f;
    const Color dielectric = EvaluateGgxBrdf(normal, normal, normal, material);
    const f32 diffuseScale = 0.96f / kPi;
    const f32 specularScale = 0.04f / (4.0f * kPi);
    CG_EXPECT_NEAR(dielectric.r, baseColor.r * diffuseScale + specularScale, 1.0e-5f);
    CG_EXPECT_NEAR(dielectric.g, baseColor.g * diffuseScale + specularScale, 1.0e-5f);
    CG_EXPECT_NEAR(dielectric.b, baseColor.b * diffuseScale + specularScale, 1.0e-5f);

    material.roughness = 0.0f;
    const Color minimumRoughness = EvaluateGgxBrdf(normal, normal, normal, material);
    CG_EXPECT_TRUE(std::isfinite(minimumRoughness.r));

    material.baseColor = Color{2.0f, 2.0f, 2.0f, 1.0f};
    material.metallic = 0.0f;
    material.roughness = 1.0f;
    const Color overbrightAlbedo = EvaluateGgxBrdf(normal, normal, normal, material);
    material.baseColor = Color{1.0f, 1.0f, 1.0f, 1.0f};
    const Color whiteAlbedo = EvaluateGgxBrdf(normal, normal, normal, material);
    CG_EXPECT_NEAR(overbrightAlbedo.r, whiteAlbedo.r, 1.0e-6f);
}

CG_TEST(software_pbr_white_furnace_does_not_gain_energy)
{
    PbrMaterial material{};
    material.baseColor = Color{1.0f, 1.0f, 1.0f, 1.0f};
    material.metallic = 0.0f;

    const f32 roughnessValues[] = {0.12f, 0.35f, 0.7f, 1.0f};
    for (const f32 roughness : roughnessValues)
    {
        material.roughness = roughness;
        const f32 reflected = IntegrateWhiteFurnace(material);
        CG_EXPECT_TRUE(reflected > 0.90f);
        CG_EXPECT_TRUE(reflected <= 1.01f);
    }
}

CG_TEST(software_pbr_normal_mapping_uses_tangent_handedness)
{
    const Vec3 normal{0.0f, 0.0f, 1.0f};
    const Vec3 tangent{1.0f, 0.0f, 0.0f};
    const Color flat{0.5f, 0.5f, 1.0f, 1.0f};
    const Vec3 unchanged = ApplyTangentSpaceNormal(normal, tangent, 1.0f, flat, 1.0f);
    CG_EXPECT_TRUE(cyber::math::ApproxEqual(unchanged, normal, 1.0e-5f));

    const Vec3 mapped =
        ApplyTangentSpaceNormal(normal, tangent, -1.0f, Color{0.5f, 1.0f, 0.5f, 1.0f}, 1.0f);
    CG_EXPECT_TRUE(mapped.y < -0.99f);
    CG_EXPECT_NEAR(cyber::math::Length(mapped), 1.0f, 1.0e-5f);
    const Vec3 degenerate = ApplyTangentSpaceNormal(normal, Vec3::Zero(), 1.0f, flat, 1.0f);
    CG_EXPECT_TRUE(cyber::math::ApproxEqual(degenerate, normal, 1.0e-5f));
}

CG_TEST(software_pbr_pixel_shader_applies_normal_map)
{
    HeapAllocator heap;
    TextureDesc textureDesc{};
    textureDesc.width = 1;
    textureDesc.height = 1;
    textureDesc.format = TextureFormat::RGBA32F;
    textureDesc.generateMips = false;
    auto textureResult = Texture::Create(textureDesc, heap);
    CG_REQUIRE(static_cast<bool>(textureResult));
    Texture normalMap = std::move(textureResult.Value());
    normalMap.SetTexel(0, 0, 0, Color{0.5f, 1.0f, 0.5f, 1.0f});

    PbrLightingData data{};
    data.eyePosition = Vec3{0.0f, 5.0f, 0.0f};
    data.ambientIrradiance = Vec3::Zero();
    data.material.baseColor = Color{1.0f, 1.0f, 1.0f, 1.0f};
    data.normalTexture = &normalMap;
    data.directional[0].direction = Vec3{0.0f, -1.0f, 0.0f};
    data.directional[0].radiance = Vec3{1.0f, 1.0f, 1.0f};
    data.directionalCount = 1;

    Varyings surface{};
    surface.normal = Vec3{0.0f, 0.0f, 1.0f};
    surface.tangent = Vec3{1.0f, 0.0f, 0.0f};
    surface.color = Color{1.0f, 1.0f, 1.0f, 1.0f};
    const Color mapped = PbrPixelShader(surface, &data);
    CG_EXPECT_TRUE(mapped.r > 0.0f);
}

CG_TEST(software_pbr_base_color_texture_uses_linear_samples)
{
    HeapAllocator heap;
    TextureDesc textureDesc{};
    textureDesc.width = 1;
    textureDesc.height = 1;
    textureDesc.format = TextureFormat::RGBA32F;
    textureDesc.generateMips = false;
    auto textureResult = Texture::Create(textureDesc, heap);
    CG_REQUIRE(static_cast<bool>(textureResult));
    Texture baseColor = std::move(textureResult.Value());
    baseColor.SetTexel(0, 0, 0, cyber::math::FromSrgb(Color{0.5f, 0.5f, 0.5f, 1.0f}));

    PbrLightingData data{};
    data.eyePosition = Vec3{0.0f, 0.0f, 5.0f};
    data.ambientIrradiance = Vec3{cyber::math::kPi, cyber::math::kPi, cyber::math::kPi};
    data.material.baseColor = Color{1.0f, 1.0f, 1.0f, 1.0f};
    data.baseColorTexture = &baseColor;

    Varyings surface{};
    surface.normal = Vec3{0.0f, 0.0f, 1.0f};
    surface.worldPosition = Vec3::Zero();
    surface.color = Color{1.0f, 1.0f, 1.0f, 1.0f};
    const Color output = PbrPixelShader(surface, &data);
    const f32 expectedLinear = cyber::math::SrgbToLinear(0.5f);
    CG_EXPECT_NEAR(output.r, expectedLinear, 1.0e-5f);
    CG_EXPECT_NEAR(output.g, expectedLinear, 1.0e-5f);
    CG_EXPECT_NEAR(output.b, expectedLinear, 1.0e-5f);
}

CG_TEST(software_pbr_directional_point_and_spot_lights)
{
    PbrLightingData data{};
    data.eyePosition = Vec3{0.0f, 5.0f, 0.0f};
    data.ambientIrradiance = Vec3::Zero();
    data.material.baseColor = Color{0.7f, 0.7f, 0.7f, 1.0f};
    data.material.roughness = 0.5f;
    data.directional[0].direction = Vec3{0.0f, -1.0f, 0.0f};
    data.directional[0].radiance = Vec3{2.0f, 2.0f, 2.0f};
    data.directionalCount = 1;
    const Varyings surface = MakeSurface();
    const Color directional = PbrPixelShader(surface, &data);
    CG_EXPECT_TRUE(directional.r > 0.0f);

    data.directionalCount = 0;
    data.pointCount = 1;
    data.points[0].position = Vec3{0.0f, 1.0f, 0.0f};
    data.points[0].intensity = Vec3{100.0f, 100.0f, 100.0f};
    data.points[0].range = 10000.0f;
    const Color nearPoint = PbrPixelShader(surface, &data);
    data.points[0].position = Vec3{0.0f, 2.0f, 0.0f};
    const Color farPoint = PbrPixelShader(surface, &data);
    CG_EXPECT_NEAR(nearPoint.r / farPoint.r, 4.0f, 0.02f);

    data.pointCount = 0;
    data.spotCount = 1;
    data.spots[0].position = Vec3{0.0f, 2.0f, 0.0f};
    data.spots[0].direction = Vec3{0.0f, -1.0f, 0.0f};
    data.spots[0].intensity = Vec3{100.0f, 100.0f, 100.0f};
    data.spots[0].range = 10.0f;
    const Color insideCone = PbrPixelShader(surface, &data);
    data.spots[0].direction = Vec3{1.0f, 0.0f, 0.0f};
    const Color outsideCone = PbrPixelShader(surface, &data);
    CG_EXPECT_TRUE(insideCone.r > 0.0f);
    CG_EXPECT_NEAR(outsideCone.r, 0.0f, 1.0e-7f);
}

CG_TEST(software_pbr_hdr_tone_mapping_is_exposed_and_bounded)
{
    const Color mapped = ToneMapAces(Color{10.0f, 1.0f, 0.1f, 0.4f});
    CG_EXPECT_TRUE(mapped.r <= 1.0f);
    CG_EXPECT_TRUE(mapped.r > mapped.g);
    CG_EXPECT_TRUE(mapped.g > mapped.b);
    CG_EXPECT_NEAR(mapped.a, 0.4f, 1.0e-7f);

    const Color brighter = ToneMapAces(Color{1.0f, 1.0f, 1.0f, 1.0f}, 1.0f);
    const Color base = ToneMapAces(Color{1.0f, 1.0f, 1.0f, 1.0f});
    CG_EXPECT_TRUE(brighter.r > base.r);
    const Color negative = ToneMapAces(Color{-1.0f, 0.0f, 0.0f, 1.0f});
    CG_EXPECT_NEAR(negative.r, 0.0f, 1.0e-7f);
    const Color extreme = ToneMapAces(Color{3.0e38f, 0.0f, 0.0f, 1.0f}, 32.0f);
    CG_EXPECT_NEAR(extreme.r, 1.0f, 1.0e-7f);
}

CG_TEST(software_pbr_shader_preserves_vertex_color_without_state)
{
    const Varyings surface = MakeSurface();
    const Color color = PbrPixelShader(surface, nullptr);
    CG_EXPECT_NEAR(color.r, surface.color.r, 1.0e-7f);
}

CG_TEST(software_pbr_nonfinite_material_does_not_leak_ambient_light)
{
    PbrLightingData data{};
    data.eyePosition = Vec3{0.0f, 5.0f, 0.0f};
    data.ambientIrradiance = Vec3{1.0f, 1.0f, 1.0f};
    data.material.roughness = std::numeric_limits<f32>::quiet_NaN();
    const Color color = PbrPixelShader(MakeSurface(), &data);
    CG_EXPECT_NEAR(color.r, 0.0f, 1.0e-7f);
    CG_EXPECT_NEAR(color.g, 0.0f, 1.0e-7f);
    CG_EXPECT_NEAR(color.b, 0.0f, 1.0e-7f);
}

CG_TEST(software_pbr_height_fog_attenuates_with_distance_and_height)
{
    PbrLightingData lighting{};
    lighting.eyePosition = Vec3::Zero();
    lighting.ambientIrradiance = Vec3{kPi, kPi, kPi};
    lighting.material.baseColor = Color{1.0f, 1.0f, 1.0f, 0.6f};
    Varyings surface = MakeSurface();
    surface.worldPosition = Vec3{0.0f, 0.0f, 10.0f};

    const Color clear = PbrPixelShader(surface, &lighting);
    lighting.fog.density = 0.1f;
    lighting.fog.scatteringAlbedo = Vec3::Zero();
    lighting.fog.integrationSteps = 16;
    const Color fogged = PbrPixelShader(surface, &lighting);
    CG_EXPECT_TRUE(fogged.r > 0.0f && fogged.r < clear.r);
    CG_EXPECT_NEAR(fogged.r, std::exp(-1.0f), 0.02f);
    CG_EXPECT_NEAR(fogged.a, 0.6f, 1.0e-6f);

    lighting.fog.heightFalloff = 1.0f;
    Varyings highSurface = surface;
    highSurface.worldPosition = Vec3{0.0f, 2.0f, 10.0f};
    Varyings lowSurface = surface;
    lowSurface.worldPosition = Vec3{0.0f, -2.0f, 10.0f};
    const Color highFog = PbrPixelShader(highSurface, &lighting);
    const Color lowFog = PbrPixelShader(lowSurface, &lighting);
    CG_EXPECT_TRUE(lowFog.r < highFog.r);
}

CG_TEST(software_pbr_fog_scattering_respects_directional_shadow_maps)
{
    HeapAllocator heap;
    auto depthResult = DepthBuffer::Create(DepthBufferDesc{4, 4, 0}, heap);
    CG_REQUIRE(static_cast<bool>(depthResult));
    DepthBuffer depth = std::move(depthResult.Value());
    depth.Clear();

    PbrLightingData lighting{};
    lighting.eyePosition = Vec3{0.0f, 0.0f, -0.5f};
    lighting.cameraForward = Vec3{0.0f, 0.0f, 1.0f};
    lighting.ambientIrradiance = Vec3::Zero();
    lighting.material.baseColor = Color{0.0f, 0.0f, 0.0f, 1.0f};
    lighting.material.metallic = 1.0f;
    lighting.directionalCount = 1;
    lighting.directional[0].direction = Vec3{0.0f, 0.0f, -1.0f};
    lighting.directional[0].radiance = Vec3{4.0f, 4.0f, 4.0f};
    lighting.directional[0].shadow.depth = &depth;
    lighting.directional[0].shadow.constantDepthBias = 0.0f;
    lighting.directional[0].shadow.slopeDepthBias = 0.0f;
    lighting.directional[0].shadow.filterRadius = 0;
    lighting.directional[0].shadow.clipFromWorld =
        cyber::math::Mat4::Translation(Vec3{0.0f, 0.0f, 0.5f});
    lighting.fog.density = 0.5f;
    lighting.fog.scatteringAlbedo = Vec3::One();
    lighting.fog.maxDistance = 2.0f;

    Varyings surface = MakeSurface();
    surface.worldPosition = Vec3{0.0f, 0.0f, 0.5f};
    CG_EXPECT_NEAR(EvaluateShadowVisibility(lighting.directional[0].shadow, Vec3{0.0f, 0.0f, 0.25f},
                                            Vec3{0.0f, 0.0f, 1.0f},
                                            Vec3{0.0f, 0.0f, 1.0f}),
                   1.0f, 1.0e-6f);
    const Color illuminatedFog = PbrPixelShader(surface, &lighting);
    depth.Clear(1.0f);
    CG_EXPECT_NEAR(EvaluateShadowVisibility(lighting.directional[0].shadow, Vec3{0.0f, 0.0f, 0.25f},
                                            Vec3{0.0f, 0.0f, 1.0f},
                                            Vec3{0.0f, 0.0f, 1.0f}),
                   0.0f, 1.0e-6f);
    const Color shadowedFog = PbrPixelShader(surface, &lighting);
    CG_EXPECT_TRUE(illuminatedFog.r > 0.05f);
    CG_EXPECT_TRUE(shadowedFog.r < illuminatedFog.r * 0.05f);
}

CG_TEST(software_pbr_henyey_greenstein_supports_backscattering)
{
    HeapAllocator heap;
    auto depthResult = DepthBuffer::Create(DepthBufferDesc{4, 4, 0}, heap);
    CG_REQUIRE(static_cast<bool>(depthResult));
    DepthBuffer depth = std::move(depthResult.Value());
    depth.Clear(0.0f);

    PbrLightingData lighting{};
    lighting.eyePosition = Vec3{0.0f, 0.0f, 1.5f};
    lighting.cameraForward = Vec3{0.0f, 0.0f, -1.0f};
    lighting.ambientIrradiance = Vec3::Zero();
    lighting.material.baseColor = Color{0.0f, 0.0f, 0.0f, 1.0f};
    lighting.material.metallic = 1.0f;
    lighting.directionalCount = 1;
    lighting.directional[0].direction = Vec3{0.0f, 0.0f, -1.0f};
    lighting.directional[0].radiance = Vec3{4.0f, 4.0f, 4.0f};
    lighting.directional[0].shadow.depth = &depth;
    lighting.directional[0].shadow.constantDepthBias = 0.0f;
    lighting.directional[0].shadow.slopeDepthBias = 0.0f;
    lighting.directional[0].shadow.filterRadius = 0;
    lighting.directional[0].shadow.clipFromWorld =
        cyber::math::Mat4::Translation(Vec3{0.0f, 0.0f, -0.5f});
    lighting.fog.density = 0.5f;
    lighting.fog.scatteringAlbedo = Vec3::One();
    lighting.fog.maxDistance = 2.0f;

    Varyings surface = MakeSurface();
    surface.worldPosition = Vec3{0.0f, 0.0f, 0.5f};
    lighting.fog.anisotropy = 0.8f;
    const Color forwardLobe = PbrPixelShader(surface, &lighting);
    lighting.fog.anisotropy = -0.8f;
    const Color backwardLobe = PbrPixelShader(surface, &lighting);
    CG_EXPECT_TRUE(std::isfinite(forwardLobe.r) && std::isfinite(backwardLobe.r));
    CG_EXPECT_TRUE(backwardLobe.r > forwardLobe.r);
}
