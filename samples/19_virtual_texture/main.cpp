// Phase 33 sample: virtual texturing.
//
// A camera flies low over a 2 km terrain whose single texture is 65,536 x
// 65,536 texels -- 16 GiB of RGBA8 at full resolution with its mips -- while
// the physical page cache holds 24 x 24 pages of 128 texels (41 MiB). The
// software renderer draws the terrain with a pixel shader that samples the
// virtual texture through its page table and writes feedback for the page it
// wanted; `VirtualTexture::Update` turns that feedback into loads on two IO
// threads and evictions from the cache.
//
// The run reports how much of each frame was drawn at the detail it asked
// for, the main-thread cost of the update, and cache traffic, and writes the
// last frame, the cache atlas and a page-residency map.
//
// Usage: cyber_sample_virtual_texture [output prefix] [--frames N]

#include <cyber/jobs/job_system.h>
#include <cyber/math/mat4.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/platform/thread.h>
#include <cyber/renderer/virtual_texture.h>
#include <cyber/software/camera.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/image_io.h>
#include <cyber/software/shader.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using cyber::core::Byte;
using cyber::core::f32;
using cyber::core::f64;
using cyber::core::i32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;
using cyber::core::usize;
using cyber::math::Color;
using cyber::math::Mat4;
using cyber::math::Vec3;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using namespace cyber::renderer;
namespace sw = cyber::software;

constexpr u32 kVirtualSize = 65536;
constexpr u32 kPageSize = 128;
constexpr u32 kBorder = 4;
constexpr u32 kCacheSide = 24;
constexpr f32 kTerrainSize = 2000.0f;  // metres; 32.8 texels per metre at mip 0
constexpr i32 kWidth = 640;
constexpr i32 kHeight = 360;
constexpr f32 kFov = 1.0f;
constexpr f64 kFrameSeconds = 1.0 / 60.0;

u32 Hash(u32 x) noexcept
{
    x ^= x >> 16u;
    x *= 0x7FEB352Du;
    x ^= x >> 15u;
    x *= 0x846CA68Bu;
    x ^= x >> 16u;
    return x;
}

// Value noise on a lattice of `cell` texels, smoothly interpolated.
f32 Noise(u32 x, u32 y, u32 cell, u32 seed) noexcept
{
    const u32 cx = x / cell;
    const u32 cy = y / cell;
    const f32 fx = static_cast<f32>(x % cell) / static_cast<f32>(cell);
    const f32 fy = static_cast<f32>(y % cell) / static_cast<f32>(cell);
    const auto corner = [seed](u32 a, u32 b)
    { return static_cast<f32>(Hash(a * 73856093u ^ b * 19349663u ^ seed) & 0xFFFFu) / 65535.0f; };
    const f32 sx = fx * fx * (3.0f - 2.0f * fx);
    const f32 sy = fy * fy * (3.0f - 2.0f * fy);
    const f32 top = corner(cx, cy) + (corner(cx + 1, cy) - corner(cx, cy)) * sx;
    const f32 bottom = corner(cx, cy + 1) + (corner(cx + 1, cy + 1) - corner(cx, cy + 1)) * sx;
    return top + (bottom - top) * sy;
}

// A procedural terrain texture, defined at mip-0 texel resolution and
// evaluated at the centre of each mip texel: fields, rock, and a survey grid
// every 512 texels whose lines are only visible at fine mips -- detail that
// shows whether a pixel got the page it asked for.
class TerrainProducer final : public VirtualPageProducer
{
public:
    bool ProducePage(u32 mip, u32 pageX, u32 pageY, Span<Byte> rgba) noexcept override
    {
        const u32 stride = kPageSize + 2u * kBorder;
        const i32 extent = static_cast<i32>(kVirtualSize >> mip);
        for (u32 j = 0; j < stride; ++j)
        {
            const i32 my = std::clamp(
                static_cast<i32>(pageY * kPageSize + j) - static_cast<i32>(kBorder), 0, extent - 1);
            for (u32 i = 0; i < stride; ++i)
            {
                const i32 mx =
                    std::clamp(static_cast<i32>(pageX * kPageSize + i) - static_cast<i32>(kBorder),
                               0, extent - 1);
                const u32 x = (static_cast<u32>(mx) << mip) + ((1u << mip) >> 1u);
                const u32 y = (static_cast<u32>(my) << mip) + ((1u << mip) >> 1u);
                const f32 field = Noise(x, y, 4096, 11u);
                const f32 detail = Noise(x, y, 96, 23u) * 0.6f + Noise(x, y, 17, 37u) * 0.4f;
                f32 r = 0.25f + 0.35f * field + 0.15f * detail;
                f32 g = 0.35f + 0.30f * (1.0f - field) + 0.15f * detail;
                f32 b = 0.15f + 0.10f * detail;
                // Grid lines are 2 mip-0 texels wide. A coarse texel covering
                // a line takes the line's share of its area, which is what a
                // box-filtered mip of the full-resolution image would hold.
                const u32 footprint = 1u << mip;
                const u32 originX = static_cast<u32>(mx) << mip;
                const u32 originY = static_cast<u32>(my) << mip;
                const bool lineX = (originX % 512u) < std::max(footprint, 2u);
                const bool lineY = (originY % 512u) < std::max(footprint, 2u);
                const f32 share = std::min(1.0f, 2.0f / static_cast<f32>(footprint));
                const f32 cover =
                    lineX || lineY ? (lineX && lineY ? std::min(1.0f, 2.0f * share) : share) : 0.0f;
                r += (0.95f - r) * cover;
                g += (0.90f - g) * cover;
                b += (0.80f - b) * cover;
                Byte* texel = rgba.Data() + (static_cast<usize>(j) * stride + i) * 4u;
                texel[0] = static_cast<Byte>(std::clamp(r, 0.0f, 1.0f) * 255.0f);
                texel[1] = static_cast<Byte>(std::clamp(g, 0.0f, 1.0f) * 255.0f);
                texel[2] = static_cast<Byte>(std::clamp(b, 0.0f, 1.0f) * 255.0f);
                texel[3] = Byte{255};
            }
        }
        return true;
    }
};

struct ShaderData
{
    sw::TransformShaderData transform{};
    VirtualTexture* vt = nullptr;
    Vec3 eye{};
    f32 texelsPerMetre = 0.0f;
    f32 focalPixels = 0.0f;
    // Per pixel: u, v and the lod it wanted (lod -1 where nothing was drawn).
    f32* pixels = nullptr;
};

void VertexShader(const sw::VertexInput& input, const void* userData,
                  sw::VertexStageOutput& output) noexcept
{
    sw::TransformVertexShader(input, &static_cast<const ShaderData*>(userData)->transform, output);
}

Color PixelShader(const sw::Varyings& varyings, const void* userData) noexcept
{
    const auto* data = static_cast<const ShaderData*>(userData);
    const Vec3 offset = varyings.worldPosition - data->eye;
    const f32 distance = std::sqrt(offset.x * offset.x + offset.y * offset.y + offset.z * offset.z);
    // Texels per pixel along the view ray: the isotropic level of detail.
    const f32 lod = std::max(std::log2(data->texelsPerMetre * distance / data->focalPixels), 0.0f);
    data->vt->RequestSample(varyings.uv.x, varyings.uv.y, lod);
    const i32 px = static_cast<i32>(varyings.pixelPosition.x);
    const i32 py = static_cast<i32>(varyings.pixelPosition.y);
    f32* record = data->pixels + (static_cast<usize>(py) * kWidth + px) * 3u;
    record[0] = varyings.uv.x;
    record[1] = varyings.uv.y;
    record[2] = lod;
    f32 rgba[4];
    data->vt->Sample(varyings.uv.x, varyings.uv.y, lod, rgba);
    return Color{rgba[0], rgba[1], rgba[2], 1.0f};
}

}  // namespace

int main(int argc, char** argv)
{
    std::string prefix;
    u32 frames = 900;
    for (int index = 1; index < argc; ++index)
    {
        if (std::strcmp(argv[index], "--frames") == 0 && index + 1 < argc)
        {
            frames = static_cast<u32>(std::strtoul(argv[++index], nullptr, 10));
        }
        else
        {
            prefix = argv[index];
        }
    }
    (void)cyber::platform::Thread::SetCurrentThreadPriority(cyber::platform::ThreadPriority::High);

    HeapAllocator allocator;
    TerrainProducer producer;
    VirtualTextureDesc desc;
    desc.width = kVirtualSize;
    desc.height = kVirtualSize;
    desc.pageSize = kPageSize;
    desc.border = kBorder;
    desc.cachePagesX = kCacheSide;
    desc.cachePagesY = kCacheSide;
    desc.ioThreads = 2;
    desc.maxInFlight = 32;
    desc.maxUploadsPerFrame = 24;
    VirtualTexture vt;
    if (!vt.Initialize(desc, producer, allocator).IsOk())
    {
        std::printf("virtual texture initialization failed\n");
        return 1;
    }
    const f64 virtualBytes = static_cast<f64>(kVirtualSize) * kVirtualSize * 4.0 * 4.0 / 3.0;
    const f64 cacheBytes = static_cast<f64>(vt.Atlas().Size());
    std::printf("virtual texture: %u x %u texels, %u mips, %llu pages (%.1f GiB with mips); cache "
                "%u pages (%.1f MiB)\n",
                kVirtualSize, kVirtualSize, vt.MipCount(),
                static_cast<unsigned long long>(vt.Stats().virtualPages),
                virtualBytes / (1u << 30u), kCacheSide * kCacheSide, cacheBytes / (1u << 20u));

    cyber::jobs::JobSystem jobs;
    cyber::jobs::JobSystem::Desc jobDesc{};
    auto createdJobs = cyber::jobs::JobSystem::Create(jobDesc, allocator);
    if (createdJobs)
    {
        jobs = std::move(createdJobs.Value());
    }
    sw::FramebufferDesc colourDesc{};
    colourDesc.width = kWidth;
    colourDesc.height = kHeight;
    colourDesc.format = sw::PixelFormat::RGBA8Unorm;
    sw::Framebuffer colour = sw::Framebuffer::Create(colourDesc, allocator).Value();
    sw::DepthBufferDesc depthDesc{};
    depthDesc.width = kWidth;
    depthDesc.height = kHeight;
    sw::DepthBuffer depth = sw::DepthBuffer::Create(depthDesc, allocator).Value();

    const f32 half = kTerrainSize * 0.5f;
    sw::VertexInput corners[4];
    const f32 positions[4][2] = {{-half, -half}, {half, -half}, {half, half}, {-half, half}};
    for (u32 i = 0; i < 4; ++i)
    {
        corners[i].position = Vec3{positions[i][0], 0.0f, positions[i][1]};
        corners[i].normal = Vec3{0.0f, 1.0f, 0.0f};
        corners[i].uv = cyber::math::Vec2{(positions[i][0] + half) / kTerrainSize,
                                          (positions[i][1] + half) / kTerrainSize};
    }
    const sw::VertexInput quad[6] = {corners[0], corners[2], corners[1],
                                     corners[0], corners[3], corners[2]};

    std::vector<f32> pixels(static_cast<usize>(kWidth) * kHeight * 3u, -1.0f);
    ShaderData data{};
    data.vt = &vt;
    data.texelsPerMetre = static_cast<f32>(kVirtualSize) / kTerrainSize;
    data.focalPixels = static_cast<f32>(kHeight) / (2.0f * std::tan(kFov * 0.5f));
    data.pixels = pixels.data();

    std::vector<f64> updateMs;
    u64 drawn = 0;
    u64 exact = 0;
    u64 withinOne = 0;
    u64 measuredFrom = frames / 3;  // after the first third: steady flight
    auto next = std::chrono::steady_clock::now();
    for (u32 frame = 0; frame < frames; ++frame)
    {
        const f32 t = static_cast<f32>(frame) * static_cast<f32>(kFrameSeconds);
        sw::Camera camera{};
        // A curving flight at 40 m/s, 25 m above the ground, looking ahead.
        camera.position = Vec3{-600.0f + 40.0f * t, 25.0f, 300.0f * std::sin(t * 0.15f)};
        camera.yawRadians = -1.5708f + 0.35f * std::cos(t * 0.15f);
        camera.pitchRadians = -0.30f;
        camera.fovYRadians = kFov;
        camera.nearPlane = 0.5f;
        camera.farPlane = 4000.0f;
        const Mat4 view = sw::CameraView(camera);
        const Mat4 projection =
            sw::CameraProjection(camera, static_cast<f32>(kWidth) / static_cast<f32>(kHeight));
        data.transform.modelViewProjection = projection * view;
        data.transform.modelToWorld = Mat4::Identity();
        data.transform.normalToWorld = Mat4::Identity();
        data.transform.eyePosition = camera.position;
        data.eye = camera.position;

        colour.Clear(Color{0.55f, 0.70f, 0.90f, 1.0f});
        depth.Clear(0.0f);
        std::fill(pixels.begin(), pixels.end(), -1.0f);
        sw::ShaderDesc shader{};
        shader.pipeline.viewport =
            sw::Viewport{0.0f, 0.0f, static_cast<f32>(kWidth), static_cast<f32>(kHeight)};
        shader.pipeline.cullMode = sw::CullMode::None;
        shader.vertexShader = &VertexShader;
        shader.pixelShader = &PixelShader;
        shader.userData = &data;
        (void)sw::DrawShadedTriangles(shader, Span<const sw::VertexInput>(quad, 6), colour, depth,
                                      jobs.IsRunning() ? &jobs : nullptr, allocator, nullptr);

        // How the frame was drawn, judged against what each pixel asked for
        // and what the page table could give it while drawing.
        if (frame >= measuredFrom)
        {
            for (i32 y = 0; y < kHeight; y += 2)
            {
                for (i32 x = 0; x < kWidth; x += 2)
                {
                    const f32* record = pixels.data() + (static_cast<usize>(y) * kWidth + x) * 3u;
                    const f32 lod = record[2];
                    if (lod < 0.0f)
                    {
                        continue;
                    }
                    ++drawn;
                    const u32 want = std::min(static_cast<u32>(lod), vt.MipCount() - 1u);
                    const u32 got = vt.Resolve(record[0], record[1], lod).mip;
                    exact += got == want ? 1u : 0u;
                    withinOne += got <= want + 1u ? 1u : 0u;
                }
            }
        }

        const auto start = std::chrono::steady_clock::now();
        (void)vt.Update();
        updateMs.push_back(
            std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - start)
                .count());

        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<f64>(kFrameSeconds));
        std::this_thread::sleep_until(next);
    }

    std::vector<f64> sorted = updateMs;
    std::sort(sorted.begin(), sorted.end());
    f64 total = 0.0;
    for (const f64 value : sorted)
    {
        total += value;
    }
    const VirtualTextureStats& stats = vt.Stats();
    std::printf("frames %u: update mean %.3f ms, p99 %.3f ms, max %.3f ms\n", frames,
                total / static_cast<f64>(sorted.size()),
                sorted[static_cast<usize>(static_cast<f64>(sorted.size() - 1) * 0.99)],
                sorted.back());
    std::printf("cache: %u of %u pages resident, %llu uploads, %llu evictions\n",
                stats.residentPages, stats.cachePages,
                static_cast<unsigned long long>(stats.totalUploads),
                static_cast<unsigned long long>(stats.totalEvictions));
    const f64 exactShare =
        drawn > 0 ? 100.0 * static_cast<f64>(exact) / static_cast<f64>(drawn) : 0.0;
    const f64 nearShare =
        drawn > 0 ? 100.0 * static_cast<f64>(withinOne) / static_cast<f64>(drawn) : 0.0;
    std::printf(
        "detail over the last %u frames: %.1f%% of pixels at the mip they asked for, %.1f%% "
        "within one mip\n",
        frames - static_cast<u32>(measuredFrom), exactShare, nearShare);

    sw::ImageWriteDesc write{};
    write.encodeSrgb = false;
    (void)sw::WriteImage((prefix + "virtual_texture_frame.ppm").c_str(), colour, allocator, write);
    // The cache atlas as an image.
    sw::FramebufferDesc atlasDesc{};
    atlasDesc.width = static_cast<i32>(vt.AtlasWidth());
    atlasDesc.height = static_cast<i32>(vt.AtlasHeight());
    atlasDesc.format = sw::PixelFormat::RGBA8Unorm;
    sw::Framebuffer atlas = sw::Framebuffer::Create(atlasDesc, allocator).Value();
    const Span<const Byte> atlasBytes = vt.Atlas();
    for (i32 y = 0; y < atlasDesc.height; ++y)
    {
        std::memcpy(atlas.Row(y), atlasBytes.Data() + static_cast<usize>(y) * atlasDesc.width * 4u,
                    static_cast<usize>(atlasDesc.width) * 4u);
    }
    (void)sw::WriteImage((prefix + "virtual_texture_cache.ppm").c_str(), atlas, allocator, write);
    // Residency of mip 2 (128 x 128 pages): bright where the page itself is
    // resident, darker the coarser the page it falls back to.
    const u32 mapMip = 2;
    sw::FramebufferDesc mapDesc{};
    mapDesc.width = static_cast<i32>(vt.PagesX(mapMip));
    mapDesc.height = static_cast<i32>(vt.PagesY(mapMip));
    mapDesc.format = sw::PixelFormat::RGBA8Unorm;
    sw::Framebuffer map = sw::Framebuffer::Create(mapDesc, allocator).Value();
    const Span<const u32> table = vt.PageTable(mapMip);
    for (i32 y = 0; y < mapDesc.height; ++y)
    {
        for (i32 x = 0; x < mapDesc.width; ++x)
        {
            const u32 resident =
                DecodePageTableEntry(table[static_cast<usize>(y) * mapDesc.width + x]).mip;
            const f32 level = 1.0f - static_cast<f32>(resident - mapMip) /
                                         static_cast<f32>(vt.MipCount() - mapMip);
            map.ClearRect(x, y, 1, 1,
                          Color{0.1f + 0.9f * level * level, 0.2f + 0.6f * level, 0.25f, 1.0f});
        }
    }
    (void)sw::WriteImage((prefix + "virtual_texture_residency.ppm").c_str(), map, allocator, write);

    const bool pass = nearShare >= 95.0 && sorted.back() < 2.0;
    if (pass)
    {
        std::printf("PASS: frames drawn from a cache %.0fx smaller than the texture\n",
                    virtualBytes / cacheBytes);
    }
    else
    {
        std::printf("FAIL: see the numbers above\n");
    }
    vt.Shutdown();
    return pass ? 0 : 1;
}
