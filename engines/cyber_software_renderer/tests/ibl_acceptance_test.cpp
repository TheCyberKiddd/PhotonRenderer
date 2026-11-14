// Phase 29 acceptance suite.
//
// The roadmap's bar for image-based lighting is behavioural: a metallic sphere
// reflects its environment, probes blend smoothly as the camera moves, and a
// white environment lights every smooth sphere white. Each measurement below
// is paired with a control configuration that fails the same threshold, so a
// passing number means the test discriminates rather than that the threshold
// is loose.

#include <cyber/jobs/job_system.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/software/ibl.h>
#include <cyber/software/pbr.h>
#include <cyber/software/shader.h>
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
using cyber::core::usize;
using cyber::math::Color;
using cyber::math::Mat4;
using cyber::math::Vec2;
using cyber::math::Vec3;

namespace
{

constexpr f32 kPi = cyber::math::kPi;
constexpr i32 kImageSize = 96;
constexpr i32 kEnvironmentFaceSize = 32;
constexpr f32 kSphereRadius = 1.0f;

struct ViewSetup
{
    Vec3 eye{0.0f, 0.0f, 4.0f};
    Vec3 target{};
    f32 fovRadians = 0.7f;
    f32 nearPlane = 0.1f;
    f32 farPlane = 100.0f;
};

Mat4 ViewFromCamera(const ViewSetup& camera)
{
    return Mat4::LookAt(camera.eye, camera.target, Vec3{0.0f, 1.0f, 0.0f});
}

Mat4 ProjectionFromCamera(const ViewSetup& camera)
{
    return Mat4::PerspectiveReverseZ(camera.fovRadians, 1.0f, camera.nearPlane, camera.farPlane);
}

// Ray through a pixel center, obtained by unprojecting through the inverse of
// the very matrix the draw used. Deriving it from a hand-built camera basis
// instead would only test that the basis matches the rasterizer's conventions,
// which is not what this suite is measuring.
Vec3 PrimaryRay(const ViewSetup& camera, i32 x, i32 y)
{
    const Mat4 clipFromWorld = ProjectionFromCamera(camera) * ViewFromCamera(camera);
    Mat4 worldFromClip = Mat4::Identity();
    if (!clipFromWorld.TryInverse(worldFromClip))
    {
        return Vec3{0.0f, 0.0f, -1.0f};
    }
    // The rasterizer maps NDC onto the viewport with a y flip, so the same
    // mapping is inverted here.
    const f32 ndcX = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(kImageSize) * 2.0f - 1.0f;
    const f32 ndcY = 1.0f - (static_cast<f32>(y) + 0.5f) / static_cast<f32>(kImageSize) * 2.0f;
    // Reverse-Z: 1 is the near plane and 0 is the far plane.
    const Vec3 nearPoint = worldFromClip.TransformPoint(Vec3{ndcX, ndcY, 1.0f});
    const Vec3 farPoint = worldFromClip.TransformPoint(Vec3{ndcX, ndcY, 0.0f});
    return cyber::math::Normalize(farPoint - nearPoint);
}

bool IntersectSphere(const Vec3& origin, const Vec3& direction, f32 radius, f32& distance)
{
    const f32 b = cyber::math::Dot(origin, direction);
    const f32 c = cyber::math::LengthSquared(origin) - radius * radius;
    const f32 discriminant = b * b - c;
    if (discriminant <= 0.0f)
    {
        return false;
    }
    distance = -b - std::sqrt(discriminant);
    return distance > 0.0f;
}

void AppendTriangle(DynamicArray<VertexInput>& vertices, const Vec3& a, const Vec3& b,
                    const Vec3& c, const Vec3& normal, const Color& color)
{
    VertexInput vertex{};
    vertex.normal = normal;
    vertex.color = color;
    vertex.position = a;
    (void)vertices.TryPushBack(vertex);
    vertex.position = b;
    (void)vertices.TryPushBack(vertex);
    vertex.position = c;
    (void)vertices.TryPushBack(vertex);
}

// Outward-facing UV sphere, so back-face culling keeps the outside.
bool BuildSphere(DynamicArray<VertexInput>& vertices, f32 radius, i32 segments, i32 rings)
{
    for (i32 ring = 0; ring < rings; ++ring)
    {
        const f32 theta0 = static_cast<f32>(ring) / static_cast<f32>(rings) * kPi;
        const f32 theta1 = static_cast<f32>(ring + 1) / static_cast<f32>(rings) * kPi;
        for (i32 segment = 0; segment < segments; ++segment)
        {
            const f32 phi0 = static_cast<f32>(segment) / static_cast<f32>(segments) * 2.0f * kPi;
            const f32 phi1 =
                static_cast<f32>(segment + 1) / static_cast<f32>(segments) * 2.0f * kPi;

            const Vec3 normals[4] = {Vec3{std::sin(theta0) * std::cos(phi0), std::cos(theta0),
                                          std::sin(theta0) * std::sin(phi0)},
                                     Vec3{std::sin(theta1) * std::cos(phi0), std::cos(theta1),
                                          std::sin(theta1) * std::sin(phi0)},
                                     Vec3{std::sin(theta1) * std::cos(phi1), std::cos(theta1),
                                          std::sin(theta1) * std::sin(phi1)},
                                     Vec3{std::sin(theta0) * std::cos(phi1), std::cos(theta0),
                                          std::sin(theta0) * std::sin(phi1)}};

            VertexInput corners[4]{};
            for (usize index = 0; index < 4; ++index)
            {
                corners[index].normal = normals[index];
                corners[index].position = normals[index] * radius;
                corners[index].color = Color{1.0f, 1.0f, 1.0f, 1.0f};
            }

            if (!vertices.TryPushBack(corners[0]) || !vertices.TryPushBack(corners[2]) ||
                !vertices.TryPushBack(corners[1]) || !vertices.TryPushBack(corners[0]) ||
                !vertices.TryPushBack(corners[3]) || !vertices.TryPushBack(corners[2]))
            {
                return false;
            }
        }
    }
    return true;
}

Cubemap MakeCubemap(Allocator& allocator, i32 faceSize)
{
    CubemapDesc desc{};
    desc.faceSize = faceSize;
    auto created = Cubemap::Create(desc, allocator);
    if (created.IsError())
    {
        return Cubemap{};
    }
    return std::move(created).Value();
}

// Six strongly different faces with a vertical gradient, so a wrong lookup
// direction is visible rather than averaged away.
void FillDirectionalEnvironment(Cubemap& cubemap)
{
    const Vec3 faceColors[kCubeFaceCount] = {Vec3{1.0f, 0.25f, 0.25f}, Vec3{0.25f, 1.0f, 0.25f},
                                             Vec3{0.25f, 0.25f, 1.0f}, Vec3{1.0f, 1.0f, 0.25f},
                                             Vec3{1.0f, 0.25f, 1.0f},  Vec3{0.25f, 1.0f, 1.0f}};
    const i32 size = cubemap.FaceSize();
    for (u32 face = 0; face < kCubeFaceCount; ++face)
    {
        for (i32 y = 0; y < size; ++y)
        {
            for (i32 x = 0; x < size; ++x)
            {
                const f32 u = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(size);
                const f32 v = (static_cast<f32>(y) + 0.5f) / static_cast<f32>(size);
                const Vec3 direction = CubeDirectionFromFaceUv(face, u, v);
                const f32 gradient = 0.4f + 0.6f * (0.5f + 0.5f * direction.y);
                const Vec3 radiance = faceColors[face] * gradient;
                cubemap.SetTexel(face, 0, x, y, Color{radiance.x, radiance.y, radiance.z, 1.0f});
            }
        }
    }
    cubemap.GenerateMips();
}

// The maps own their storage and the lighting struct only borrows it, so the
// descriptor is built on demand from the current addresses rather than stored.
// A smooth directional environment: continuous across every cube edge, so a
// comparison against it measures the lighting rather than the few degrees of
// normal error a tessellated sphere carries.
void FillSmoothEnvironment(Cubemap& cubemap)
{
    const i32 size = cubemap.FaceSize();
    for (u32 face = 0; face < kCubeFaceCount; ++face)
    {
        for (i32 y = 0; y < size; ++y)
        {
            for (i32 x = 0; x < size; ++x)
            {
                const f32 u = (static_cast<f32>(x) + 0.5f) / static_cast<f32>(size);
                const f32 v = (static_cast<f32>(y) + 0.5f) / static_cast<f32>(size);
                const Vec3 direction = CubeDirectionFromFaceUv(face, u, v);
                cubemap.SetTexel(face, 0, x, y,
                                 Color{0.5f + 0.4f * direction.x, 0.5f + 0.4f * direction.y,
                                       0.5f + 0.4f * direction.z, 1.0f});
            }
        }
    }
    cubemap.GenerateMips();
}

enum class EnvironmentKind
{
    Constant,
    SharpFaces,
    Smooth,
};

struct Environment
{
    Cubemap source;
    Cubemap prefiltered;
    Cubemap irradiance;
    Texture brdfLut;
    bool valid = false;

    [[nodiscard]] EnvironmentLighting Lighting() const
    {
        EnvironmentLighting lighting{};
        lighting.specular = &prefiltered;
        lighting.diffuseIrradiance = &irradiance;
        lighting.brdfLut = &brdfLut;
        return lighting;
    }
};

bool BuildEnvironment(Allocator& allocator, const Color& constantRadiance, EnvironmentKind kind,
                      Environment& environment)
{
    environment.source = MakeCubemap(allocator, kEnvironmentFaceSize);
    environment.prefiltered = MakeCubemap(allocator, kEnvironmentFaceSize);
    environment.irradiance = MakeCubemap(allocator, 16);
    if (!environment.source.IsValid() || !environment.prefiltered.IsValid() ||
        !environment.irradiance.IsValid())
    {
        return false;
    }

    switch (kind)
    {
    case EnvironmentKind::Constant:
        environment.source.Clear(constantRadiance);
        environment.source.GenerateMips();
        break;
    case EnvironmentKind::SharpFaces:
        FillDirectionalEnvironment(environment.source);
        break;
    case EnvironmentKind::Smooth:
        FillSmoothEnvironment(environment.source);
        break;
    }

    SpecularPrefilterDesc prefilter{};
    prefilter.sampleCount = 128;
    prefilter.useJobs = false;
    if (PrefilterSpecularCubemap(environment.source, prefilter, environment.prefiltered).IsError())
    {
        return false;
    }

    IrradianceFilterDesc filter{};
    filter.useJobs = false;
    if (BuildDiffuseIrradianceCubemap(environment.source, filter, environment.irradiance).IsError())
    {
        return false;
    }

    BrdfLutDesc lutDesc{};
    lutDesc.size = 64;
    lutDesc.sampleCount = 512;
    lutDesc.useJobs = false;
    if (BuildSplitSumBrdfLut(lutDesc, environment.brdfLut, allocator).IsError())
    {
        return false;
    }

    environment.valid = true;
    return true;
}

struct RenderTargets
{
    Framebuffer color;
    DepthBuffer depth;
    bool valid = false;
};

RenderTargets MakeTargets(Allocator& allocator, i32 width, i32 height)
{
    RenderTargets targets{};
    FramebufferDesc colorDesc{};
    colorDesc.width = width;
    colorDesc.height = height;
    colorDesc.format = PixelFormat::RGBA32F;
    auto color = Framebuffer::Create(colorDesc, allocator);
    if (color.IsError())
    {
        return targets;
    }
    targets.color = std::move(color).Value();

    DepthBufferDesc depthDesc{};
    depthDesc.width = width;
    depthDesc.height = height;
    auto depth = DepthBuffer::Create(depthDesc, allocator);
    if (depth.IsError())
    {
        return targets;
    }
    targets.depth = std::move(depth).Value();
    targets.valid = true;
    return targets;
}

// One pointer reaches both stages, so the scene bundles the transform and the
// lighting and the wrapper shaders hand each stage its own half.
struct SceneUniforms
{
    TransformShaderData transform{};
    PbrLightingData lighting{};
};

void SceneVertex(const VertexInput& input, const void* userData, VertexStageOutput& output) noexcept
{
    TransformVertexShader(input, &static_cast<const SceneUniforms*>(userData)->transform, output);
}

Color ScenePixel(const Varyings& varyings, const void* userData) noexcept
{
    return PbrPixelShader(varyings, &static_cast<const SceneUniforms*>(userData)->lighting);
}

// Unlit pass used by the screen-space reflection scene: the vertex colors are
// the surface radiance the trace is expected to find.
Color VertexColorPixel(const Varyings& varyings, const void* userData) noexcept
{
    (void)userData;
    return varyings.color;
}

bool RenderPbr(const ViewSetup& camera, SceneUniforms& uniforms, Span<const VertexInput> vertices,
               Framebuffer& color, DepthBuffer& depth, Allocator& allocator)
{
    const Mat4 view = ViewFromCamera(camera);
    const Mat4 projection = ProjectionFromCamera(camera);

    uniforms.transform.modelViewProjection = projection * view;
    uniforms.transform.modelToWorld = Mat4::Identity();
    uniforms.transform.normalToWorld = Mat4::Identity();
    uniforms.transform.eyePosition = camera.eye;
    uniforms.lighting.eyePosition = camera.eye;
    uniforms.lighting.cameraForward = cyber::math::Normalize(camera.target - camera.eye);

    ShaderDesc desc{};
    desc.pipeline.view = view;
    desc.pipeline.projection = projection;
    desc.pipeline.viewport =
        Viewport{0.0f, 0.0f, static_cast<f32>(color.Width()), static_cast<f32>(color.Height())};
    desc.pipeline.cullMode = CullMode::Back;
    desc.pipeline.useJobs = false;
    desc.vertexShader = &SceneVertex;
    desc.pixelShader = &ScenePixel;
    desc.userData = &uniforms;

    color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    depth.Clear(0.0f);

    return DrawShadedTriangles(desc, vertices, color, depth, nullptr, allocator).IsOk();
}

bool RenderPbr(const ViewSetup& camera, SceneUniforms& uniforms, Span<const VertexInput> vertices,
               RenderTargets& targets, Allocator& allocator)
{
    return RenderPbr(camera, uniforms, vertices, targets.color, targets.depth, allocator);
}

}  // namespace

CG_TEST(ibl_acceptance_metallic_sphere_reflects_its_environment)
{
    HeapAllocator allocator;
    Environment environment;
    CG_REQUIRE(BuildEnvironment(allocator, Color{}, EnvironmentKind::SharpFaces, environment));
    const EnvironmentLighting environmentLighting = environment.Lighting();

    ViewSetup camera{};
    camera.eye = Vec3{0.0f, 0.0f, 4.0f};

    // A mirror metal sphere: the outgoing radiance is the environment radiance
    // along the reflected view ray, which is what makes this measurable
    // without a reference renderer.
    f32 errorSum = 0.0f;
    f32 worstError = 0.0f;
    u32 covered = 0;
    for (i32 y = 0; y < kImageSize; ++y)
    {
        for (i32 x = 0; x < kImageSize; ++x)
        {
            const Vec3 ray = PrimaryRay(camera, x, y);
            f32 distance = 0.0f;
            if (!IntersectSphere(camera.eye, ray, kSphereRadius, distance))
            {
                continue;
            }
            const Vec3 position = camera.eye + ray * distance;
            const Vec3 normal = cyber::math::Normalize(position);
            const Vec3 view = -ray;
            const Vec3 reflected = cyber::math::Reflect(ray, normal);

            IblSurface surface{};
            surface.normal = normal;
            surface.viewDirection = view;
            surface.baseColor = Vec3::One();
            surface.metallic = 1.0f;
            surface.roughness = 0.0f;
            const Vec3 evaluated = EvaluateImageBasedLighting(environmentLighting, surface);

            const Color reference = SampleCubemapFace(environment.source, reflected, 0);
            const f32 error =
                (std::abs(evaluated.x - reference.r) + std::abs(evaluated.y - reference.g) +
                 std::abs(evaluated.z - reference.b)) /
                3.0f;
            errorSum += error;
            worstError = std::max(worstError, error);
            ++covered;
        }
    }

    CG_REQUIRE(covered > 2000);
    const f32 meanError = errorSum / static_cast<f32>(covered);
    std::fprintf(stderr, "mirror sphere: mean %.5f worst %.5f\n", static_cast<double>(meanError),
                 static_cast<double>(worstError));
    // A mirror lookup through the prefiltered chain reproduces the environment
    // to within the resampling error of a 32-texel face.
    CG_EXPECT_TRUE(meanError < 0.02f);
    CG_EXPECT_TRUE(worstError < 0.15f);

    // Control: the same measurement with the irradiance map standing in for
    // the prefiltered chain. It is the same radiometric quantity scaled by pi
    // and carries no directional detail, so it must fail the threshold.
    EnvironmentLighting wrong = environmentLighting;
    wrong.specular = &environment.irradiance;
    f32 controlSum = 0.0f;
    u32 controlCovered = 0;
    for (i32 y = 0; y < kImageSize; y += 2)
    {
        for (i32 x = 0; x < kImageSize; x += 2)
        {
            const Vec3 ray = PrimaryRay(camera, x, y);
            f32 distance = 0.0f;
            if (!IntersectSphere(camera.eye, ray, kSphereRadius, distance))
            {
                continue;
            }
            const Vec3 position = camera.eye + ray * distance;
            const Vec3 normal = cyber::math::Normalize(position);
            const Vec3 reflected = cyber::math::Reflect(ray, normal);

            IblSurface surface{};
            surface.normal = normal;
            surface.viewDirection = -ray;
            surface.baseColor = Vec3::One();
            surface.metallic = 1.0f;
            surface.roughness = 0.0f;
            const Vec3 evaluated = EvaluateImageBasedLighting(wrong, surface);
            const Color reference = SampleCubemapFace(environment.source, reflected, 0);
            controlSum +=
                (std::abs(evaluated.x - reference.r) + std::abs(evaluated.y - reference.g) +
                 std::abs(evaluated.z - reference.b)) /
                3.0f;
            ++controlCovered;
        }
    }
    CG_REQUIRE(controlCovered > 0);
    std::fprintf(stderr, "control mean %.5f\n",
                 static_cast<double>(controlSum / static_cast<f32>(controlCovered)));
    CG_EXPECT_TRUE(controlSum / static_cast<f32>(controlCovered) > 0.02f);
}

CG_TEST(ibl_acceptance_rendered_probe_sphere_matches_the_analytic_reflection)
{
    HeapAllocator allocator;
    Environment environment;
    CG_REQUIRE(BuildEnvironment(allocator, Color{}, EnvironmentKind::Smooth, environment));
    const EnvironmentLighting environmentLighting = environment.Lighting();

    RenderTargets targets = MakeTargets(allocator, kImageSize, kImageSize);
    CG_REQUIRE(targets.valid);

    DynamicArray<VertexInput> vertices(allocator);
    CG_REQUIRE(BuildSphere(vertices, kSphereRadius, 64, 32));

    ViewSetup camera{};
    camera.eye = Vec3{0.0f, 0.0f, 4.0f};

    ReflectionProbe probe{};
    probe.influenceMin = Vec3{-2.0f, -2.0f, -2.0f};
    probe.influenceMax = Vec3{2.0f, 2.0f, 2.0f};
    probe.environment = environmentLighting;
    const ReflectionProbe probes[] = {probe};

    SceneUniforms uniforms{};
    uniforms.lighting.reflectionProbes.probes = Span<const ReflectionProbe>{probes, 1};
    uniforms.lighting.material.baseColor = Color{1.0f, 1.0f, 1.0f, 1.0f};
    uniforms.lighting.material.metallic = 1.0f;
    uniforms.lighting.material.roughness = 0.0f;
    uniforms.lighting.directionalCount = 0;
    CG_REQUIRE(RenderPbr(camera, uniforms,
                         Span<const VertexInput>{vertices.Data(), vertices.Size()}, targets,
                         allocator));

    // Compare the rendered image with the analytic reflection of the same
    // sphere. Only pixels well inside the silhouette are compared, because a
    // tessellated sphere and an exact one disagree at the rim.
    f32 errorSum = 0.0f;
    u32 covered = 0;
    for (i32 y = 0; y < kImageSize; ++y)
    {
        for (i32 x = 0; x < kImageSize; ++x)
        {
            const Vec3 ray = PrimaryRay(camera, x, y);
            f32 distance = 0.0f;
            if (!IntersectSphere(camera.eye, ray, kSphereRadius * 0.9f, distance))
            {
                continue;
            }
            f32 exactDistance = 0.0f;
            CG_REQUIRE(IntersectSphere(camera.eye, ray, kSphereRadius, exactDistance));
            const Vec3 position = camera.eye + ray * exactDistance;
            const Vec3 normal = cyber::math::Normalize(position);
            const Vec3 reflected = cyber::math::Reflect(ray, normal);
            const Color reference = SampleCubemapFace(environment.source, reflected, 0);
            const Color rendered = targets.color.GetPixel(x, y);
            errorSum += (std::abs(rendered.r - reference.r) + std::abs(rendered.g - reference.g) +
                         std::abs(rendered.b - reference.b)) /
                        3.0f;
            ++covered;
        }
    }

    CG_REQUIRE(covered > 1000);
    const f32 meanError = errorSum / static_cast<f32>(covered);
    std::fprintf(stderr, "rendered sphere: mean %.5f over %u pixels\n",
                 static_cast<double>(meanError), covered);
    {
        u32 nonBlack = 0;
        for (i32 y = 0; y < kImageSize; ++y)
        {
            for (i32 x = 0; x < kImageSize; ++x)
            {
                if (targets.color.GetPixel(x, y).r > 0.001f)
                {
                    ++nonBlack;
                }
            }
        }
        const i32 mid = kImageSize / 2;
        const Vec3 ray = PrimaryRay(camera, mid, mid);
        f32 hitDistance = 0.0f;
        (void)IntersectSphere(camera.eye, ray, kSphereRadius, hitDistance);
        const Vec3 position = camera.eye + ray * hitDistance;
        const Vec3 reflected = cyber::math::Reflect(ray, cyber::math::Normalize(position));
        const Color reference = SampleCubemapFace(environment.source, reflected, 0);
        const Color rendered = targets.color.GetPixel(mid, mid);
        std::fprintf(stderr,
                     "  nonBlack=%u center rendered=(%.3f %.3f %.3f) reference=(%.3f %.3f %.3f)\n",
                     nonBlack, static_cast<double>(rendered.r), static_cast<double>(rendered.g),
                     static_cast<double>(rendered.b), static_cast<double>(reference.r),
                     static_cast<double>(reference.g), static_cast<double>(reference.b));
        const Color offCenter = targets.color.GetPixel(mid + 20, mid);
        const Vec3 offRay = PrimaryRay(camera, mid + 20, mid);
        f32 offDistance = 0.0f;
        (void)IntersectSphere(camera.eye, offRay, kSphereRadius, offDistance);
        const Vec3 offPosition = camera.eye + offRay * offDistance;
        const Color offReference =
            SampleCubemapFace(environment.source,
                              cyber::math::Reflect(offRay, cyber::math::Normalize(offPosition)), 0);
        std::fprintf(stderr, "  +20x rendered=(%.3f %.3f %.3f) reference=(%.3f %.3f %.3f)\n",
                     static_cast<double>(offCenter.r), static_cast<double>(offCenter.g),
                     static_cast<double>(offCenter.b), static_cast<double>(offReference.r),
                     static_cast<double>(offReference.g), static_cast<double>(offReference.b));
    }
    // The sphere is tessellated and its normals are interpolated, so the
    // tolerance is wider than the analytic measurement above.
    CG_EXPECT_TRUE(meanError < 0.05f);
}

CG_TEST(ibl_acceptance_white_environment_renders_white_spheres)
{
    HeapAllocator allocator;
    constexpr f32 kRadiance = 0.5f;
    Environment environment;
    CG_REQUIRE(BuildEnvironment(allocator, Color{kRadiance, kRadiance, kRadiance, 1.0f},
                                EnvironmentKind::Constant, environment));
    const EnvironmentLighting environmentLighting = environment.Lighting();

    RenderTargets targets = MakeTargets(allocator, kImageSize, kImageSize);
    CG_REQUIRE(targets.valid);

    DynamicArray<VertexInput> vertices(allocator);
    CG_REQUIRE(BuildSphere(vertices, kSphereRadius, 48, 24));

    ViewSetup camera{};
    camera.eye = Vec3{0.0f, 0.0f, 4.0f};

    for (i32 step = 0; step <= 4; ++step)
    {
        const f32 roughness = static_cast<f32>(step) / 4.0f;
        for (i32 metal = 0; metal <= 1; ++metal)
        {
            SceneUniforms uniforms{};
            uniforms.lighting.environment = &environmentLighting;
            uniforms.lighting.material.baseColor = Color{1.0f, 1.0f, 1.0f, 1.0f};
            uniforms.lighting.material.metallic = static_cast<f32>(metal);
            uniforms.lighting.material.roughness = roughness;
            CG_REQUIRE(RenderPbr(camera, uniforms,
                                 Span<const VertexInput>{vertices.Data(), vertices.Size()}, targets,
                                 allocator));

            f32 worst = 0.0f;
            u32 covered = 0;
            for (i32 y = 0; y < kImageSize; ++y)
            {
                for (i32 x = 0; x < kImageSize; ++x)
                {
                    const Vec3 ray = PrimaryRay(camera, x, y);
                    f32 distance = 0.0f;
                    if (!IntersectSphere(camera.eye, ray, kSphereRadius * 0.85f, distance))
                    {
                        continue;
                    }
                    const Color rendered = targets.color.GetPixel(x, y);
                    worst = std::max(worst, std::abs(rendered.r - kRadiance));
                    ++covered;
                }
            }
            CG_REQUIRE(covered > 500);
            std::fprintf(stderr, "furnace roughness %.2f metal %d worst %.5f\n",
                         static_cast<double>(roughness), metal, static_cast<double>(worst));
            // A white furnace: a fully reflective white surface returns the
            // environment radiance whatever its roughness or metalness.
            CG_REQUIRE(worst < 0.04f);
        }
    }

    // Control: the flat ambient term the renderer uses without image-based
    // lighting cannot light a metal at all, so the same scene renders black.
    SceneUniforms flat{};
    flat.lighting.ambientIrradiance = Vec3{kPi * kRadiance, kPi * kRadiance, kPi * kRadiance};
    flat.lighting.material.baseColor = Color{1.0f, 1.0f, 1.0f, 1.0f};
    flat.lighting.material.metallic = 1.0f;
    flat.lighting.material.roughness = 0.3f;
    CG_REQUIRE(RenderPbr(camera, flat, Span<const VertexInput>{vertices.Data(), vertices.Size()},
                         targets, allocator));
    const Color center = targets.color.GetPixel(kImageSize / 2, kImageSize / 2);
    CG_EXPECT_TRUE(std::abs(center.r - kRadiance) > 0.04f);
}

CG_TEST(ibl_acceptance_probes_blend_smoothly_along_a_camera_path)
{
    HeapAllocator allocator;
    Cubemap first = MakeCubemap(allocator, 8);
    Cubemap second = MakeCubemap(allocator, 8);
    CG_REQUIRE(first.IsValid() && second.IsValid());
    first.Clear(Color{kPi, 0.2f * kPi, 0.2f * kPi, 1.0f});
    second.Clear(Color{0.2f * kPi, 0.2f * kPi, kPi, 1.0f});

    EnvironmentLighting fallbackLighting{};
    Cubemap fallbackMap = MakeCubemap(allocator, 8);
    CG_REQUIRE(fallbackMap.IsValid());
    fallbackMap.Clear(Color{0.1f * kPi, kPi, 0.1f * kPi, 1.0f});
    fallbackLighting.diffuseIrradiance = &fallbackMap;

    ReflectionProbe probes[2]{};
    probes[0].influenceMin = Vec3{-6.0f, -2.0f, -2.0f};
    probes[0].influenceMax = Vec3{0.5f, 2.0f, 2.0f};
    probes[0].blendDistance = 1.5f;
    probes[0].environment.diffuseIrradiance = &first;
    probes[1].influenceMin = Vec3{-0.5f, -2.0f, -2.0f};
    probes[1].influenceMax = Vec3{6.0f, 2.0f, 2.0f};
    probes[1].blendDistance = 1.5f;
    probes[1].environment.diffuseIrradiance = &second;

    ReflectionProbeSet set{};
    set.probes = Span<const ReflectionProbe>{probes, 2};
    set.fallback = &fallbackLighting;

    IblSurface surface{};
    surface.normal = Vec3{0.0f, 1.0f, 0.0f};
    surface.viewDirection = Vec3{0.0f, 1.0f, 0.0f};
    surface.baseColor = Vec3::One();
    surface.metallic = 0.0f;
    surface.roughness = 1.0f;

    constexpr u32 kSteps = 400;
    constexpr f32 kStart = -8.0f;
    constexpr f32 kEnd = 8.0f;
    const f32 stepSize = (kEnd - kStart) / static_cast<f32>(kSteps);

    f32 worstStep = 0.0f;
    Vec3 previous{};
    bool havePrevious = false;
    for (u32 step = 0; step <= kSteps; ++step)
    {
        const Vec3 position{kStart + stepSize * static_cast<f32>(step), 0.0f, 0.0f};
        const Vec3 current = EvaluateReflectionProbes(set, position, surface);
        if (havePrevious)
        {
            const f32 change = std::max(
                std::max(std::abs(current.x - previous.x), std::abs(current.y - previous.y)),
                std::abs(current.z - previous.z));
            worstStep = std::max(worstStep, change);
        }
        previous = current;
        havePrevious = true;
    }

    std::fprintf(stderr, "probe path: worst step %.5f\n", static_cast<double>(worstStep));
    // The probes differ by 0.8 in a channel, and the path crosses the whole
    // blend region in 4 cm steps, so a smooth blend keeps every step far below
    // a tenth of that contrast.
    CG_EXPECT_TRUE(worstStep < 0.05f);

    // Control: without a blend distance the same path crosses a hard boundary
    // and the step change jumps by an order of magnitude.
    ReflectionProbe hard[2] = {probes[0], probes[1]};
    hard[0].blendDistance = 0.0f;
    hard[1].blendDistance = 0.0f;
    ReflectionProbeSet hardSet = set;
    hardSet.probes = Span<const ReflectionProbe>{hard, 2};

    f32 worstHardStep = 0.0f;
    havePrevious = false;
    for (u32 step = 0; step <= kSteps; ++step)
    {
        const Vec3 position{kStart + stepSize * static_cast<f32>(step), 0.0f, 0.0f};
        const Vec3 current = EvaluateReflectionProbes(hardSet, position, surface);
        if (havePrevious)
        {
            const f32 change = std::max(
                std::max(std::abs(current.x - previous.x), std::abs(current.y - previous.y)),
                std::abs(current.z - previous.z));
            worstHardStep = std::max(worstHardStep, change);
        }
        previous = current;
        havePrevious = true;
    }
    std::fprintf(stderr, "probe path without blending: worst step %.5f\n",
                 static_cast<double>(worstHardStep));
    CG_EXPECT_TRUE(worstHardStep > 0.1f);
    CG_EXPECT_TRUE(worstHardStep > worstStep * 5.0f);
}

CG_TEST(ibl_acceptance_screen_space_reflection_finds_the_mirrored_surface)
{
    HeapAllocator allocator;
    constexpr i32 kSize = 160;
    RenderTargets targets = MakeTargets(allocator, kSize, kSize);
    CG_REQUIRE(targets.valid);

    // A dark floor at y = 0 and a bright panel floating above it. The panel's
    // reflection is what the trace must find.
    DynamicArray<VertexInput> vertices(allocator);
    const Color floorColor{0.05f, 0.05f, 0.05f, 1.0f};
    const Color panelColor{0.9f, 0.3f, 0.1f, 1.0f};
    AppendTriangle(vertices, Vec3{-8.0f, 0.0f, -8.0f}, Vec3{-8.0f, 0.0f, 8.0f},
                   Vec3{8.0f, 0.0f, 8.0f}, Vec3{0.0f, 1.0f, 0.0f}, floorColor);
    AppendTriangle(vertices, Vec3{-8.0f, 0.0f, -8.0f}, Vec3{8.0f, 0.0f, 8.0f},
                   Vec3{8.0f, 0.0f, -8.0f}, Vec3{0.0f, 1.0f, 0.0f}, floorColor);
    AppendTriangle(vertices, Vec3{-2.0f, 2.0f, -1.0f}, Vec3{-2.0f, 2.0f, 1.0f},
                   Vec3{2.0f, 2.0f, 1.0f}, Vec3{0.0f, -1.0f, 0.0f}, panelColor);
    AppendTriangle(vertices, Vec3{-2.0f, 2.0f, -1.0f}, Vec3{2.0f, 2.0f, 1.0f},
                   Vec3{2.0f, 2.0f, -1.0f}, Vec3{0.0f, -1.0f, 0.0f}, panelColor);

    ViewSetup camera{};
    camera.eye = Vec3{0.0f, 1.2f, 7.0f};
    camera.target = Vec3{0.0f, 0.8f, 0.0f};
    camera.fovRadians = 0.9f;
    camera.nearPlane = 0.1f;
    camera.farPlane = 60.0f;

    const Mat4 view = ViewFromCamera(camera);
    const Mat4 projection =
        Mat4::PerspectiveReverseZ(camera.fovRadians, 1.0f, camera.nearPlane, camera.farPlane);

    TransformShaderData transform{};
    transform.modelViewProjection = projection * view;
    transform.modelToWorld = Mat4::Identity();
    transform.normalToWorld = Mat4::Identity();
    transform.eyePosition = camera.eye;

    // Vertex colors carry the surface radiance, so the pass needs no lighting.
    ShaderDesc desc{};
    desc.pipeline.view = view;
    desc.pipeline.projection = projection;
    desc.pipeline.viewport = Viewport{0.0f, 0.0f, static_cast<f32>(kSize), static_cast<f32>(kSize)};
    desc.pipeline.cullMode = CullMode::None;
    desc.pipeline.useJobs = false;
    desc.vertexShader = &TransformVertexShader;
    desc.pixelShader = &VertexColorPixel;
    desc.userData = &transform;

    targets.color.Clear(Color{0.0f, 0.0f, 0.0f, 1.0f});
    targets.depth.Clear(0.0f);
    CG_REQUIRE(DrawShadedTriangles(desc, Span<const VertexInput>{vertices.Data(), vertices.Size()},
                                   targets.color, targets.depth, nullptr, allocator)
                   .IsOk());

    auto built = HiZPyramid::Build(targets.depth, allocator);
    CG_REQUIRE(built.IsOk());
    HiZPyramid pyramid = std::move(built).Value();

    ScreenSpaceReflectionDesc ssr{};
    ssr.color = &targets.color;
    ssr.depth = &targets.depth;
    ssr.clipFromWorld = projection * view;
    ssr.perspectiveProjection = true;
    ssr.nearPlane = camera.nearPlane;
    ssr.farPlane = camera.farPlane;
    ssr.thickness = 0.5f;
    ssr.maxWorldDistance = 20.0f;
    ssr.maxSteps = 160;
    ssr.edgeFadeFraction = 0.05f;

    // Floor points directly under the panel see it in a mirror reflection.
    u32 hits = 0;
    u32 matches = 0;
    u32 linearSteps = 0;
    u32 hiZSteps = 0;
    // The mirror image of the panel sits at y = -2, so the floor points that
    // see it are the ones where the eye ray to that image crosses y = 0, which
    // is a band in front of the panel rather than under it.
    for (i32 index = 0; index < 9; ++index)
    {
        const f32 z = 4.05f + 0.075f * static_cast<f32>(index);
        const Vec3 position{0.0f, 0.0f, z};
        const Vec3 toEye = cyber::math::Normalize(camera.eye - position);
        const Vec3 mirror = cyber::math::Reflect(-toEye, Vec3{0.0f, 1.0f, 0.0f});

        const ScreenSpaceReflectionSample linear =
            TraceScreenSpaceReflection(ssr, position, mirror);
        ssr.hiZ = &pyramid;
        const ScreenSpaceReflectionSample accelerated =
            TraceScreenSpaceReflection(ssr, position, mirror);
        ssr.hiZ = nullptr;

        linearSteps += linear.steps;
        hiZSteps += accelerated.steps;
        std::fprintf(stderr,
                     "ssr z=%.2f hit=%d steps=%u rgb=(%.3f %.3f %.3f) conf=%.3f | hiz hit=%d "
                     "steps=%u r=%.3f\n",
                     static_cast<double>(z), linear.hit ? 1 : 0, linear.steps,
                     static_cast<double>(linear.radiance.x), static_cast<double>(linear.radiance.y),
                     static_cast<double>(linear.radiance.z), static_cast<double>(linear.confidence),
                     accelerated.hit ? 1 : 0, accelerated.steps,
                     static_cast<double>(accelerated.radiance.x));
        if (!linear.hit)
        {
            continue;
        }
        ++hits;
        // Both paths must find the same surface, and it must be the panel.
        if (linear.radiance.x > 0.5f && accelerated.hit && accelerated.radiance.x > 0.5f)
        {
            ++matches;
        }
    }

    CG_EXPECT_TRUE(hits >= 7);
    CG_EXPECT_TRUE(matches >= 7);
    // The pyramid exists to lower the cost, so it must not raise it.
    CG_EXPECT_TRUE(hiZSteps <= linearSteps);

    // A ray that leaves the screen has no answer and says so rather than
    // returning the nearest border pixel.
    const Vec3 outward = cyber::math::Normalize(Vec3{0.0f, 0.2f, 1.0f});
    const ScreenSpaceReflectionSample escaping =
        TraceScreenSpaceReflection(ssr, Vec3{0.0f, 0.0f, 6.0f}, outward);
    CG_EXPECT_TRUE(!escaping.hit || escaping.confidence < 1.0f);
}
