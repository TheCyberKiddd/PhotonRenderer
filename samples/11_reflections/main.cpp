#include <cyber/core/result.h>
#include <cyber/math/mat4.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/antialias.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/ibl.h>
#include <cyber/software/image_io.h>
#include <cyber/software/pbr.h>
#include <cyber/software/shader.h>
#include <cyber/software/shadow.h>
#include <cyber/software/texture.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

using namespace cyber::memory;
using namespace cyber::software;
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

constexpr i32 kWidth = 800;
constexpr i32 kHeight = 600;
constexpr i32 kProbeFaceSize = 128;
constexpr i32 kShadowMapSize = 1024;
constexpr Vec3 kCubeCenter{0.0f, 1.19f, 0.35f};
constexpr Vec3 kCubeHalfExtents{0.76f, 0.76f, 0.76f};
constexpr f32 kPi = cyber::math::kPi;

struct SceneUniforms
{
    TransformShaderData transform{};
    PbrLightingData lighting{};
};

struct CaptureScene
{
    const VertexInput* vertices = nullptr;
    usize vertexCount = 0;
    const VertexInput* platformVertices = nullptr;
    usize platformVertexCount = 0;
    const Texture* platformTexture = nullptr;
    const PbrDirectionalLight* sun = nullptr;
    HeapAllocator* allocator = nullptr;
};

void SceneVertex(const VertexInput& input, const void* userData, VertexStageOutput& output) noexcept
{
    TransformVertexShader(input, &static_cast<const SceneUniforms*>(userData)->transform, output);
}

Color ScenePixel(const Varyings& varyings, const void* userData) noexcept
{
    return PbrPixelShader(varyings, &static_cast<const SceneUniforms*>(userData)->lighting);
}

bool PushTriangle(DynamicArray<VertexInput>& mesh, const Vec3& a, const Vec3& b, const Vec3& c,
                  const Vec3& normal, const Color& color)
{
    VertexInput vertex{};
    vertex.normal = normal;
    vertex.color = color;
    vertex.position = a;
    if (!mesh.TryPushBack(vertex))
    {
        return false;
    }
    vertex.position = b;
    if (!mesh.TryPushBack(vertex))
    {
        return false;
    }
    vertex.position = c;
    return mesh.TryPushBack(vertex);
}

bool PushSmoothTriangle(DynamicArray<VertexInput>& mesh, const Vec3& a, const Vec3& normalA,
                        const Vec3& b, const Vec3& normalB, const Vec3& c, const Vec3& normalC,
                        const Color& color)
{
    VertexInput vertex{};
    vertex.color = color;
    vertex.position = a;
    vertex.normal = normalA;
    if (!mesh.TryPushBack(vertex))
    {
        return false;
    }
    vertex.position = b;
    vertex.normal = normalB;
    if (!mesh.TryPushBack(vertex))
    {
        return false;
    }
    vertex.position = c;
    vertex.normal = normalC;
    return mesh.TryPushBack(vertex);
}

bool PushQuad(DynamicArray<VertexInput>& mesh, const Vec3& a, const Vec3& b, const Vec3& c,
              const Vec3& d, const Vec3& normal, const Color& color)
{
    return PushTriangle(mesh, a, b, c, normal, color) && PushTriangle(mesh, a, c, d, normal, color);
}

bool AddBox(DynamicArray<VertexInput>& mesh, const Vec3& center, const Vec3& halfExtents,
            const Color& color, f32 rotationY = 0.0f);

bool PushTexturedQuad(DynamicArray<VertexInput>& mesh, const Vec3& a, const Vec2& uvA,
                      const Vec3& b, const Vec2& uvB, const Vec3& c, const Vec2& uvC, const Vec3& d,
                      const Vec2& uvD, const Vec3& normal, const Color& color)
{
    const auto push = [&](const Vec3& position, const Vec2& uv)
    {
        VertexInput vertex{};
        vertex.position = position;
        vertex.normal = normal;
        vertex.uv = uv;
        vertex.color = color;
        return mesh.TryPushBack(vertex);
    };
    return push(a, uvA) && push(b, uvB) && push(c, uvC) && push(a, uvA) && push(c, uvC) &&
           push(d, uvD);
}

bool BuildPlatform(DynamicArray<VertexInput>& mesh)
{
    constexpr f32 centerX = 0.0f;
    constexpr f32 centerZ = 0.35f;
    constexpr f32 halfWidth = 2.25f;
    constexpr f32 halfDepth = 1.9f;
    constexpr f32 top = 0.425f;
    if (!AddBox(mesh, Vec3{centerX, 0.24f, centerZ}, Vec3{halfWidth, 0.18f, halfDepth},
                Color{0.24f, 0.29f, 0.36f, 1.0f}))
    {
        return false;
    }

    return PushTexturedQuad(mesh, Vec3{centerX - halfWidth, top, centerZ - halfDepth},
                            Vec2{0.0f, 0.0f}, Vec3{centerX - halfWidth, top, centerZ + halfDepth},
                            Vec2{0.0f, 3.0f}, Vec3{centerX + halfWidth, top, centerZ + halfDepth},
                            Vec2{4.0f, 3.0f}, Vec3{centerX + halfWidth, top, centerZ - halfDepth},
                            Vec2{4.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f},
                            Color{1.0f, 1.0f, 1.0f, 1.0f});
}

void BuildPlatformTexture(Texture& texture)
{
    constexpr i32 size = 128;
    constexpr i32 tileSize = 32;
    for (i32 y = 0; y < size; ++y)
    {
        for (i32 x = 0; x < size; ++x)
        {
            const i32 tileX = x / tileSize;
            const i32 tileY = y / tileSize;
            const i32 localX = x % tileSize;
            const i32 localY = y % tileSize;
            const i32 variation = (x * 17 + y * 29 + tileX * 11 + tileY * 7) % 13;
            Color texel{};
            if (localX < 2 || localY < 2)
            {
                texel = Color{0.035f, 0.045f, 0.06f, 1.0f};
            }
            else
            {
                const f32 grain = static_cast<f32>(variation) * 0.003f;
                const bool alternate = ((tileX + tileY) & 1) != 0;
                const Color base =
                    alternate ? Color{0.12f, 0.30f, 0.43f, 1.0f} : Color{0.46f, 0.25f, 0.11f, 1.0f};
                texel = Color{base.r + grain, base.g + grain, base.b + grain, 1.0f};
            }
            texture.SetTexel(0, x, y, texel);
        }
    }
    texture.GenerateMips();
}

bool BuildFloor(DynamicArray<VertexInput>& mesh)
{
    constexpr i32 kTiles = 16;
    constexpr f32 kExtent = 6.0f;
    constexpr f32 kStep = (2.0f * kExtent) / static_cast<f32>(kTiles);
    const Color colors[2] = {Color{0.34f, 0.37f, 0.42f, 1.0f}, Color{0.075f, 0.095f, 0.13f, 1.0f}};
    const Vec3 up{0.0f, 1.0f, 0.0f};
    for (i32 z = 0; z < kTiles; ++z)
    {
        for (i32 x = 0; x < kTiles; ++x)
        {
            const f32 x0 = -kExtent + static_cast<f32>(x) * kStep;
            const f32 z0 = -kExtent + static_cast<f32>(z) * kStep;
            const f32 x1 = x0 + kStep;
            const f32 z1 = z0 + kStep;
            const Color color = colors[(x + z) & 1];
            if (!PushQuad(mesh, Vec3{x0, 0.0f, z0}, Vec3{x0, 0.0f, z1}, Vec3{x1, 0.0f, z1},
                          Vec3{x1, 0.0f, z0}, up, color))
            {
                return false;
            }
        }
    }
    return true;
}

bool AddBox(DynamicArray<VertexInput>& mesh, const Vec3& center, const Vec3& halfExtents,
            const Color& color, f32 rotationY)
{
    const Vec3 normals[6] = {Vec3{0.0f, 0.0f, 1.0f}, Vec3{0.0f, 0.0f, -1.0f},
                             Vec3{1.0f, 0.0f, 0.0f}, Vec3{-1.0f, 0.0f, 0.0f},
                             Vec3{0.0f, 1.0f, 0.0f}, Vec3{0.0f, -1.0f, 0.0f}};
    const Vec3 tangents[6] = {Vec3{1.0f, 0.0f, 0.0f},  Vec3{-1.0f, 0.0f, 0.0f},
                              Vec3{0.0f, 0.0f, -1.0f}, Vec3{0.0f, 0.0f, 1.0f},
                              Vec3{1.0f, 0.0f, 0.0f},  Vec3{1.0f, 0.0f, 0.0f}};
    const Vec3 bitangents[6] = {Vec3{0.0f, 1.0f, 0.0f},  Vec3{0.0f, 1.0f, 0.0f},
                                Vec3{0.0f, 1.0f, 0.0f},  Vec3{0.0f, 1.0f, 0.0f},
                                Vec3{0.0f, 0.0f, -1.0f}, Vec3{0.0f, 0.0f, 1.0f}};
    const f32 sine = std::sin(rotationY);
    const f32 cosine = std::cos(rotationY);
    const auto rotate = [&](const Vec3& value) {
        return Vec3{cosine * value.x + sine * value.z, value.y, -sine * value.x + cosine * value.z};
    };

    for (usize face = 0; face < 6; ++face)
    {
        const Vec3 normal = rotate(normals[face]);
        const f32 tangentExtent = face < 2 ? halfExtents.x : halfExtents.z;
        const f32 bitangentExtent = face < 4 ? halfExtents.y : halfExtents.z;
        const Vec3 faceCenter =
            center +
            rotate(normals[face] *
                   (face < 2 ? halfExtents.z : (face < 4 ? halfExtents.x : halfExtents.y)));
        const Vec3 right = rotate(tangents[face] * tangentExtent);
        const Vec3 up = rotate(bitangents[face] * bitangentExtent);
        if (!PushQuad(mesh, faceCenter - right - up, faceCenter + right - up,
                      faceCenter + right + up, faceCenter - right + up, normal, color))
        {
            return false;
        }
    }
    return true;
}

bool AddSphere(DynamicArray<VertexInput>& mesh, const Vec3& center, f32 radius, const Color& color,
               u32 segments = 32, u32 rings = 20)
{
    for (u32 ring = 0; ring < rings; ++ring)
    {
        const f32 theta0 = static_cast<f32>(ring) / static_cast<f32>(rings) * kPi;
        const f32 theta1 = static_cast<f32>(ring + 1) / static_cast<f32>(rings) * kPi;
        for (u32 segment = 0; segment < segments; ++segment)
        {
            const f32 phi0 = static_cast<f32>(segment) / static_cast<f32>(segments) * 2.0f * kPi;
            const f32 phi1 =
                static_cast<f32>(segment + 1) / static_cast<f32>(segments) * 2.0f * kPi;
            const auto normalAt = [](f32 theta, f32 phi) {
                return Vec3{std::sin(theta) * std::cos(phi), std::cos(theta),
                            std::sin(theta) * std::sin(phi)};
            };
            const Vec3 n00 = normalAt(theta0, phi0);
            const Vec3 n01 = normalAt(theta0, phi1);
            const Vec3 n10 = normalAt(theta1, phi0);
            const Vec3 n11 = normalAt(theta1, phi1);
            if (!PushSmoothTriangle(mesh, center + n00 * radius, n00, center + n10 * radius, n10,
                                    center + n11 * radius, n11, color) ||
                !PushSmoothTriangle(mesh, center + n00 * radius, n00, center + n11 * radius, n11,
                                    center + n01 * radius, n01, color))
            {
                return false;
            }
        }
    }
    return true;
}

bool BuildRoom(DynamicArray<VertexInput>& mesh)
{
    if (!BuildFloor(mesh))
    {
        return false;
    }
    const Color backWall{0.11f, 0.15f, 0.22f, 1.0f};
    const Color sideWall{0.16f, 0.19f, 0.25f, 1.0f};
    const Color ceiling{0.25f, 0.27f, 0.30f, 1.0f};
    if (!PushQuad(mesh, Vec3{-6.0f, 0.0f, -6.0f}, Vec3{6.0f, 0.0f, -6.0f}, Vec3{6.0f, 6.0f, -6.0f},
                  Vec3{-6.0f, 6.0f, -6.0f}, Vec3{0.0f, 0.0f, 1.0f}, backWall) ||
        !PushQuad(mesh, Vec3{-6.0f, 0.0f, 6.0f}, Vec3{-6.0f, 6.0f, 6.0f}, Vec3{6.0f, 6.0f, 6.0f},
                  Vec3{6.0f, 0.0f, 6.0f}, Vec3{0.0f, 0.0f, -1.0f}, sideWall) ||
        !PushQuad(mesh, Vec3{-6.0f, 0.0f, -6.0f}, Vec3{-6.0f, 6.0f, -6.0f}, Vec3{-6.0f, 6.0f, 6.0f},
                  Vec3{-6.0f, 0.0f, 6.0f}, Vec3{1.0f, 0.0f, 0.0f}, sideWall) ||
        !PushQuad(mesh, Vec3{6.0f, 0.0f, 6.0f}, Vec3{6.0f, 6.0f, 6.0f}, Vec3{6.0f, 6.0f, -6.0f},
                  Vec3{6.0f, 0.0f, -6.0f}, Vec3{-1.0f, 0.0f, 0.0f}, sideWall) ||
        !PushQuad(mesh, Vec3{-6.0f, 6.0f, 6.0f}, Vec3{-6.0f, 6.0f, -6.0f}, Vec3{6.0f, 6.0f, -6.0f},
                  Vec3{6.0f, 6.0f, 6.0f}, Vec3{0.0f, -1.0f, 0.0f}, ceiling))
    {
        return false;
    }

    return AddBox(mesh, Vec3{-2.15f, 0.58f, -1.25f}, Vec3{0.58f, 0.58f, 0.58f},
                  Color{0.035f, 0.23f, 0.82f, 1.0f}, -0.22f) &&
           AddBox(mesh, Vec3{1.85f, 0.48f, -1.75f}, Vec3{0.78f, 0.48f, 0.62f},
                  Color{0.82f, 0.12f, 0.035f, 1.0f}, 0.18f) &&
           AddBox(mesh, Vec3{0.15f, 0.28f, -3.1f}, Vec3{0.48f, 0.28f, 0.48f},
                  Color{0.08f, 0.62f, 0.35f, 1.0f}, 0.12f) &&
           AddBox(mesh, Vec3{2.9f, 0.52f, 0.7f}, Vec3{0.55f, 0.52f, 0.55f},
                  Color{0.95f, 0.42f, 0.035f, 1.0f}, 0.16f) &&
           AddSphere(mesh, Vec3{-3.35f, 0.4f, -2.7f}, 0.4f, Color{0.68f, 0.08f, 0.47f, 1.0f}) &&
           AddBox(mesh, Vec3{-3.7f, 2.1f, -5.88f}, Vec3{0.95f, 1.2f, 0.07f},
                  Color{0.75f, 0.34f, 0.07f, 1.0f}) &&
           AddBox(mesh, Vec3{2.9f, 2.4f, -5.88f}, Vec3{0.72f, 1.0f, 0.07f},
                  Color{0.07f, 0.49f, 0.72f, 1.0f});
}

bool AppendCasters(const DynamicArray<VertexInput>& scene,
                   const DynamicArray<VertexInput>& platform, const DynamicArray<VertexInput>& cube,
                   DynamicArray<VertexInput>& casters)
{
    const DynamicArray<VertexInput>* meshes[] = {&scene, &platform, &cube};
    for (const DynamicArray<VertexInput>* mesh : meshes)
    {
        for (usize index = 0; index < mesh->Size(); ++index)
        {
            if (!casters.TryPushBack((*mesh)[index]))
            {
                return false;
            }
        }
    }
    return true;
}

bool RenderGeometry(Span<const VertexInput> vertices, const Mat4& model,
                    const ReflectionProbeSet& probes, const EnvironmentLighting* environment,
                    const PbrDirectionalLight& sun, const PbrMaterial& material, const Vec3& eye,
                    const Vec3& target, Framebuffer& color, DepthBuffer& depth,
                    HeapAllocator& allocator, const Texture* baseColorTexture = nullptr)
{
    const Mat4 view = Mat4::LookAt(eye, target, Vec3{0.0f, 1.0f, 0.0f});
    const Mat4 projection = Mat4::PerspectiveReverseZ(
        0.9f, static_cast<f32>(color.Width()) / static_cast<f32>(color.Height()), 0.1f, 60.0f);
    SceneUniforms uniforms{};
    uniforms.transform.modelViewProjection = projection * view * model;
    uniforms.transform.modelToWorld = model;
    uniforms.transform.normalToWorld = Mat4::Identity();
    uniforms.transform.eyePosition = eye;
    uniforms.lighting.eyePosition = eye;
    uniforms.lighting.cameraForward = cyber::math::Normalize(target - eye);
    uniforms.lighting.ambientIrradiance = Vec3{0.18f, 0.20f, 0.24f};
    uniforms.lighting.reflectionProbes = probes;
    uniforms.lighting.environment = environment;
    uniforms.lighting.material = material;
    uniforms.lighting.baseColorTexture = baseColorTexture;
    uniforms.lighting.sampler.addressU = AddressMode::Repeat;
    uniforms.lighting.sampler.addressV = AddressMode::Repeat;
    uniforms.lighting.directional[0] = sun;
    uniforms.lighting.directional[1].direction = cyber::math::Normalize(Vec3{0.38f, -0.62f, 0.72f});
    uniforms.lighting.directional[1].radiance = Vec3{1.25f, 1.45f, 1.75f};
    uniforms.lighting.directionalCount = 2;
    ShaderDesc shader{};
    shader.pipeline.view = view;
    shader.pipeline.projection = projection;
    shader.pipeline.viewport =
        Viewport{0.0f, 0.0f, static_cast<f32>(color.Width()), static_cast<f32>(color.Height())};
    shader.pipeline.cullMode = CullMode::None;
    shader.pipeline.useJobs = false;
    shader.vertexShader = &SceneVertex;
    shader.pixelShader = &ScenePixel;
    shader.userData = &uniforms;
    return DrawShadedTriangles(shader, vertices, color, depth, nullptr, allocator).IsOk();
}

cyber::core::Status<cyber::core::Error> CaptureRoomFace(const ReflectionProbeCaptureView& view,
                                                        Framebuffer& color, DepthBuffer& depth,
                                                        void* userData) noexcept
{
    auto& scene = *static_cast<CaptureScene*>(userData);
    const Vec3 forward = CubeDirectionFromFaceUv(static_cast<u32>(view.face), 0.5f, 0.5f);
    ShaderDesc shader{};
    shader.pipeline.view = view.view;
    shader.pipeline.projection = view.projection;
    shader.pipeline.viewport =
        Viewport{0.0f, 0.0f, static_cast<f32>(color.Width()), static_cast<f32>(color.Height())};
    shader.pipeline.cullMode = CullMode::None;
    shader.pipeline.useJobs = false;
    shader.vertexShader = &SceneVertex;
    shader.pixelShader = &ScenePixel;

    const auto draw =
        [&](const VertexInput* vertices, usize vertexCount, const Texture* texture) noexcept
    {
        SceneUniforms uniforms{};
        uniforms.transform.modelViewProjection = view.projection * view.view;
        uniforms.transform.modelToWorld = Mat4::Identity();
        uniforms.transform.normalToWorld = Mat4::Identity();
        uniforms.transform.eyePosition = view.position;
        uniforms.lighting.eyePosition = view.position;
        uniforms.lighting.cameraForward = forward;
        uniforms.lighting.ambientIrradiance = Vec3{0.18f, 0.20f, 0.24f};
        uniforms.lighting.material.baseColor = Color{1.0f, 1.0f, 1.0f, 1.0f};
        uniforms.lighting.material.metallic = 0.08f;
        uniforms.lighting.material.roughness = 0.42f;
        uniforms.lighting.baseColorTexture = texture;
        uniforms.lighting.sampler.addressU = AddressMode::Repeat;
        uniforms.lighting.sampler.addressV = AddressMode::Repeat;
        uniforms.lighting.directional[0] = *scene.sun;
        uniforms.lighting.directional[1].direction =
            cyber::math::Normalize(Vec3{0.38f, -0.62f, 0.72f});
        uniforms.lighting.directional[1].radiance = Vec3{1.25f, 1.45f, 1.75f};
        uniforms.lighting.directionalCount = 2;
        shader.userData = &uniforms;
        return DrawShadedTriangles(shader, Span<const VertexInput>{vertices, vertexCount}, color,
                                   depth, nullptr, *scene.allocator)
            .IsOk();
    };

    if (!draw(scene.vertices, scene.vertexCount, nullptr) ||
        !draw(scene.platformVertices, scene.platformVertexCount, scene.platformTexture))
    {
        return cyber::core::Status<cyber::core::Error>::Err(cyber::core::Error{
            cyber::core::ErrorCode::Internal, "failed to render reflection probe face"});
    }
    return cyber::core::Status<cyber::core::Error>::Ok();
}

void ToneMap(Framebuffer& image)
{
    for (i32 y = 0; y < image.Height(); ++y)
    {
        for (i32 x = 0; x < image.Width(); ++x)
        {
            image.SetPixel(x, y, ToneMapAces(image.GetPixel(x, y), 0.25f));
        }
    }
}

}  // namespace

int main(int argc, char** argv)
{
    const char* outputPath = argc > 1 ? argv[1] : "reflections.ppm";
    if (argc > 2)
    {
        std::printf("usage: cyber_sample_reflections [output.ppm]\n");
        return 2;
    }

    HeapAllocator allocator;
    DynamicArray<VertexInput> scene(allocator);
    DynamicArray<VertexInput> platform(allocator);
    DynamicArray<VertexInput> cube(allocator);
    DynamicArray<VertexInput> casters(allocator);
    if (!BuildRoom(scene) || !BuildPlatform(platform) ||
        !AddBox(cube, kCubeCenter, kCubeHalfExtents, Color{0.92f, 0.96f, 1.0f, 1.0f}, 0.22f) ||
        !AppendCasters(scene, platform, cube, casters))
    {
        std::printf("failed to build reflection scene geometry\n");
        return 1;
    }

    TextureDesc platformTextureDesc{};
    platformTextureDesc.width = 128;
    platformTextureDesc.height = 128;
    platformTextureDesc.format = TextureFormat::RGBA32F;
    platformTextureDesc.generateMips = true;
    auto platformTextureResult = Texture::Create(platformTextureDesc, allocator);
    if (!platformTextureResult)
    {
        std::printf("failed to create platform texture: %s\n",
                    platformTextureResult.Error().message);
        return 1;
    }
    Texture platformTexture = std::move(platformTextureResult).Value();
    BuildPlatformTexture(platformTexture);

    constexpr Vec3 lightTravel = {-0.48f, -1.0f, -0.36f};
    const Vec3 lightCenter{0.0f, 1.6f, -0.5f};
    const Vec3 lightEye = lightCenter - cyber::math::Normalize(lightTravel) * 12.0f;
    const Mat4 lightView = Mat4::LookAt(lightEye, lightCenter, Vec3{0.0f, 1.0f, 0.0f});
    const Mat4 lightProjection = Mat4::OrthographicReverseZ(-8.0f, 8.0f, -8.0f, 8.0f, 0.1f, 35.0f);

    DepthBufferDesc shadowDesc{};
    shadowDesc.width = kShadowMapSize;
    shadowDesc.height = kShadowMapSize;
    auto shadowResult = DepthBuffer::Create(shadowDesc, allocator);
    if (!shadowResult)
    {
        std::printf("failed to allocate shadow map\n");
        return 1;
    }
    DepthBuffer shadowDepth = std::move(shadowResult).Value();
    ShadowRenderDesc shadowRender{};
    shadowRender.clipFromWorld = lightProjection * lightView;
    shadowRender.cullMode = CullMode::None;
    shadowRender.useJobs = false;
    if (!RenderShadowMap(shadowRender, Span<const VertexInput>{casters.Data(), casters.Size()},
                         shadowDepth, nullptr, allocator))
    {
        std::printf("failed to render scene shadow map\n");
        return 1;
    }
    PbrDirectionalLight sun{};
    sun.direction = cyber::math::Normalize(lightTravel);
    sun.radiance = Vec3{3.0f, 2.7f, 2.4f};
    sun.shadow.depth = &shadowDepth;
    sun.shadow.clipFromWorld = shadowRender.clipFromWorld;
    sun.shadow.constantDepthBias = 0.001f;
    sun.shadow.slopeDepthBias = 0.003f;
    sun.shadow.filterRadius = 2;
    sun.shadow.softFilter = true;

    ReflectionProbeCaptureDesc captureDesc{};
    captureDesc.position = kCubeCenter;
    captureDesc.faceSize = kProbeFaceSize;
    captureDesc.supersampleFactor = kDefaultSupersampleFactor;
    captureDesc.resolveFilter = kDefaultResolveFilter;
    captureDesc.nearPlane = 0.08f;
    captureDesc.farPlane = 30.0f;
    CaptureScene captureScene{scene.Data(),     scene.Size(), platform.Data(), platform.Size(),
                              &platformTexture, &sun,         &allocator};
    auto captured = CaptureReflectionProbe(captureDesc, &CaptureRoomFace, &captureScene, allocator);
    if (!captured)
    {
        std::printf("reflection probe capture failed: %s\n", captured.Error().message);
        return 1;
    }
    EnvironmentMapDesc environmentDesc{};
    environmentDesc.faceSize = kProbeFaceSize;
    environmentDesc.diffuseFaceSize = 16;
    environmentDesc.brdfLutSize = 48;
    environmentDesc.specularSampleCount = 96;
    environmentDesc.brdfSampleCount = 256;
    environmentDesc.useJobs = false;
    auto environmentResult = EnvironmentMap::Create(captured.Value(), environmentDesc, allocator);
    if (!environmentResult)
    {
        std::printf("reflection environment preprocessing failed: %s\n",
                    environmentResult.Error().message);
        return 1;
    }
    EnvironmentMap environment = std::move(environmentResult).Value();
    EnvironmentLighting environmentLighting = environment.Lighting();
    environmentLighting.intensity = 1.6f;

    ReflectionProbe probe{};
    probe.position = captureDesc.position;
    probe.influenceMin = Vec3{-5.8f, 0.05f, -5.8f};
    probe.influenceMax = Vec3{5.8f, 5.8f, 5.8f};
    probe.proxyMin = Vec3{-5.9f, 0.05f, -5.9f};
    probe.proxyMax = Vec3{5.9f, 5.9f, 5.9f};
    probe.blendDistance = 0.45f;
    probe.environment = environmentLighting;
    const ReflectionProbe probeArray[] = {probe};
    ReflectionProbeSet probes{};
    probes.probes = Span<const ReflectionProbe>{probeArray, 1};
    probes.fallback = &environmentLighting;

    AntiAliasTargetDesc antialiasDesc{};
    antialiasDesc.width = kWidth;
    antialiasDesc.height = kHeight;
    antialiasDesc.quality = kDefaultAntiAliasQuality;
    antialiasDesc.filter = kDefaultResolveFilter;
    antialiasDesc.format = PixelFormat::RGBA32F;
    auto antialiasResult = AntiAliasTarget::Create(antialiasDesc, allocator);
    if (!antialiasResult)
    {
        std::printf("failed to allocate anti-aliased scene targets: %s\n",
                    antialiasResult.Error().message);
        return 1;
    }
    AntiAliasTarget antialias = std::move(antialiasResult).Value();
    antialias.Clear(Color{0.015f, 0.021f, 0.032f, 1.0f});

    FramebufferDesc resolvedDesc{};
    resolvedDesc.width = kWidth;
    resolvedDesc.height = kHeight;
    resolvedDesc.format = PixelFormat::RGBA32F;
    auto resolvedResult = Framebuffer::Create(resolvedDesc, allocator);
    if (!resolvedResult)
    {
        std::printf("failed to allocate resolved scene target: %s\n",
                    resolvedResult.Error().message);
        return 1;
    }
    Framebuffer resolved = std::move(resolvedResult).Value();

    const Vec3 eye{3.8f, 3.6f, 5.2f};
    const Vec3 target{0.0f, 1.3f, -0.4f};
    PbrMaterial roomMaterial{};
    roomMaterial.baseColor = Color{1.0f, 1.0f, 1.0f, 1.0f};
    roomMaterial.metallic = 0.08f;
    roomMaterial.roughness = 0.42f;
    PbrMaterial cubeMaterial{};
    cubeMaterial.baseColor = Color{0.92f, 0.96f, 1.0f, 1.0f};
    cubeMaterial.metallic = 0.92f;
    cubeMaterial.roughness = 0.075f;

    if (!RenderGeometry(Span<const VertexInput>{cube.Data(), cube.Size()}, Mat4::Identity(), probes,
                        nullptr, sun, cubeMaterial, eye, target, antialias.Color(),
                        antialias.Depth(), allocator))
    {
        std::printf("failed to render reflective cube\n");
        return 1;
    }
    ReflectionProbeSet noProbes{};
    if (!RenderGeometry(Span<const VertexInput>{scene.Data(), scene.Size()}, Mat4::Identity(),
                        noProbes, &environmentLighting, sun, roomMaterial, eye, target,
                        antialias.Color(), antialias.Depth(), allocator))
    {
        std::printf("failed to render room geometry\n");
        return 1;
    }
    PbrMaterial platformMaterial = roomMaterial;
    platformMaterial.metallic = 0.05f;
    platformMaterial.roughness = 0.58f;
    if (!RenderGeometry(Span<const VertexInput>{platform.Data(), platform.Size()}, Mat4::Identity(),
                        noProbes, &environmentLighting, sun, platformMaterial, eye, target,
                        antialias.Color(), antialias.Depth(), allocator, &platformTexture))
    {
        std::printf("failed to render textured platform\n");
        return 1;
    }

    ResolveDesc resolveDesc{};
    resolveDesc.useJobs = false;
    auto resolveStatus = antialias.Resolve(resolved, nullptr, allocator, resolveDesc);
    if (!resolveStatus)
    {
        std::printf("failed to resolve anti-aliased scene: %s\n", resolveStatus.Error().message);
        return 1;
    }
    ToneMap(resolved);
    ImageWriteDesc imageDesc{};
    imageDesc.format = ImageFormat::Ppm;
    imageDesc.encodeSrgb = true;
    if (!WriteImage(outputPath, resolved, allocator, imageDesc))
    {
        std::printf("failed to write rendered sample: %s\n", outputPath);
        return 1;
    }

    std::printf("Rendered a %zu-triangle room at %dx%d with %dx%d supersampling, a textured "
                "platform, six-face probe capture, and a %d x %d shadow map.\n",
                (scene.Size() + platform.Size() + cube.Size()) / 3, kWidth, kHeight,
                antialias.Factor(), antialias.Factor(), kShadowMapSize, kShadowMapSize);
    std::printf("Reflective cube: local probe, metallic 0.92, roughness 0.075.\n");
    std::printf("Wrote %s\n", outputPath);
    return 0;
}
