// Phase 33 sample: texture streaming under a video-memory budget.
//
// A camera flies down a corridor lined with 400 wall panels, each carrying its
// own 2048x2048 BC1 texture with a full mip chain: about 1.07 GiB of texture
// data against a 64 MiB budget. Every frame the sample projects each panel,
// reports the mip it needs to the TextureStreamer, and lets the streamer load
// and evict on its background IO threads. The upload sink copies each level
// into a page pool the size of the budget, standing in for a GPU heap.
//
// The run is paced at 60 frames per second in real time, so the IO threads
// have the time a game would give them. It reports the main-thread cost of
// streaming per frame (the hitch measure), budget use, and how far the view
// lags behind what it needs, and writes residency-view images.
//
// Usage: cyber_sample_texture_streaming [output prefix] [--frames N]

#include <cyber/assets/bc_codec.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/platform/thread.h>
#include <cyber/renderer/texture_streaming.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace
{

using cyber::core::Byte;
using cyber::core::f32;
using cyber::core::f64;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;
using cyber::core::usize;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using namespace cyber::renderer;

constexpr u32 kPanelCount = 400;
constexpr u32 kTextureSize = 2048;
constexpr u64 kBudgetBytes = 64ull << 20u;
constexpr f32 kPanelSpacing = 2.0f;  // metres between panel pairs along the corridor
constexpr f32 kPanelSize = 4.0f;
constexpr f32 kCorridorHalfWidth = 2.5f;
constexpr f32 kCameraSpeed = 12.0f;  // metres per second
constexpr u32 kScreenWidth = 1920;
constexpr u32 kScreenHeight = 1080;
constexpr f32 kVerticalFov = 1.0f;  // radians
constexpr f32 kFarPlane = 400.0f;
constexpr f64 kFrameSeconds = 1.0 / 60.0;
// A frame whose streaming work on the main thread exceeds this is a hitch:
// 2 ms is an eighth of a 60 Hz frame.
constexpr f64 kHitchMilliseconds = 2.0;
// Simulated storage: per-request latency plus bandwidth, like an NVMe drive
// behind a file system.
constexpr f64 kReadLatencySeconds = 0.0004;
constexpr f64 kReadBytesPerSecond = 1.5e9;

u32 LevelSize(u32 level)
{
    return std::max(kTextureSize >> level, 1u);
}

u64 Bc1Bytes(u32 size)
{
    const u64 blocks = (size + 3u) / 4u;
    return blocks * blocks * 8u;
}

// A procedural BC1 texture: a two-colour checker whose hue identifies the
// panel and whose cell size identifies the level. Each level is two blocks
// encoded with the Phase 33 BC1 encoder and tiled, which costs the IO thread
// about what decompressing a package would.
class PanelSource final : public TextureMipSource
{
public:
    explicit PanelSource(u32 panel) noexcept : m_panel(panel)
    {
        u32 levels = 0;
        for (u32 size = kTextureSize; size >= 1; size /= 2)
        {
            ++levels;
            if (size == 1)
            {
                break;
            }
        }
        m_levels = levels;
    }

    u32 LevelCount() const noexcept override
    {
        return m_levels;
    }

    StreamingLevelInfo Level(u32 level) const noexcept override
    {
        const u32 size = LevelSize(level);
        return StreamingLevelInfo{size, size, Bc1Bytes(size)};
    }

    bool ReadLevel(u32 level, Span<Byte> destination) noexcept override
    {
        const auto start = std::chrono::steady_clock::now();
        Byte blocks[2][8];
        const f32 hue = static_cast<f32>(m_panel % 37u) / 37.0f * 6.2831853f;
        for (u32 variant = 0; variant < 2; ++variant)
        {
            const f32 shade = variant == 0 ? 1.0f : 0.35f + 0.04f * static_cast<f32>(level);
            u8 rgba[64];
            for (u32 texel = 0; texel < 16; ++texel)
            {
                rgba[texel * 4 + 0] =
                    static_cast<u8>(255.0f * shade * (0.5f + 0.5f * std::cos(hue)));
                rgba[texel * 4 + 1] =
                    static_cast<u8>(255.0f * shade * (0.5f + 0.5f * std::cos(hue - 2.1f)));
                rgba[texel * 4 + 2] =
                    static_cast<u8>(255.0f * shade * (0.5f + 0.5f * std::cos(hue + 2.1f)));
                rgba[texel * 4 + 3] = 255;
            }
            cyber::assets::EncodeBc1Block(rgba, blocks[variant], cyber::assets::BcEncodeSettings{});
        }
        const u32 blocksAcross = (LevelSize(level) + 3u) / 4u;
        const u32 cell = std::max(blocksAcross / 8u, 1u);
        for (u32 y = 0; y < blocksAcross; ++y)
        {
            for (u32 x = 0; x < blocksAcross; ++x)
            {
                const u32 variant = ((x / cell) + (y / cell)) & 1u;
                std::memcpy(destination.Data() + (static_cast<usize>(y) * blocksAcross + x) * 8u,
                            blocks[variant], 8);
            }
        }
        // Storage time the read would really take.
        const f64 seconds =
            kReadLatencySeconds + static_cast<f64>(destination.Size()) / kReadBytesPerSecond;
        std::this_thread::sleep_until(start + std::chrono::duration<f64>(seconds));
        return true;
    }

private:
    u32 m_panel = 0;
    u32 m_levels = 0;
};

// Stands in for a GPU heap: each upload is copied into the pages the
// streamer allocated, exactly the bytes a D3D12 backend would write into its
// upload ring before CopyTextureRegion.
class PoolSink final : public TextureUploadSink
{
public:
    PoolSink(u64 bytes, u32 pageBytes) : m_pool(static_cast<usize>(bytes)), m_pageBytes(pageBytes)
    {
    }

    void UploadLevels(TextureHandle, u32, u32, Span<const Byte> data,
                      Span<const u32> pages) noexcept override
    {
        usize offset = 0;
        for (usize i = 0; i < pages.Size() && offset < data.Size(); ++i)
        {
            const usize bytes = std::min<usize>(m_pageBytes, data.Size() - offset);
            const usize page = static_cast<usize>(pages[i]) * m_pageBytes;
            if (page + bytes > m_pool.size())
            {
                overflow = true;
                return;
            }
            std::memcpy(m_pool.data() + page, data.Data() + offset, bytes);
            offset += bytes;
        }
    }

    void EvictLevel(TextureHandle, u32, Span<const u32>) noexcept override {}

    bool overflow = false;

private:
    std::vector<Byte> m_pool;
    u32 m_pageBytes = 0;
};

bool WritePpm(const std::string& path, const std::vector<Byte>& rgba, u32 width, u32 height)
{
    std::ofstream file(path, std::ios::binary);
    if (!file)
    {
        return false;
    }
    file << "P6\n" << width << ' ' << height << "\n255\n";
    std::vector<char> rgb(static_cast<usize>(width) * height * 3u);
    for (usize i = 0; i < static_cast<usize>(width) * height; ++i)
    {
        rgb[i * 3 + 0] = static_cast<char>(rgba[i * 4 + 0]);
        rgb[i * 3 + 1] = static_cast<char>(rgba[i * 4 + 1]);
        rgb[i * 3 + 2] = static_cast<char>(rgba[i * 4 + 2]);
    }
    file.write(rgb.data(), static_cast<std::streamsize>(rgb.size()));
    return static_cast<bool>(file);
}

struct Panel
{
    f32 x = 0.0f;
    f32 z = 0.0f;
};

}  // namespace

int main(int argc, char** argv)
{
    std::string prefix;
    u32 frames = 1800;
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

    // Like a game's main thread: above the background IO threads.
    (void)cyber::platform::Thread::SetCurrentThreadPriority(cyber::platform::ThreadPriority::High);
    HeapAllocator allocator;
    TextureStreamerDesc desc;
    desc.budgetBytes = kBudgetBytes;
    desc.ioThreads = 2;
    desc.maxInFlight = 24;
    // One top mip per frame: uploads are where streaming touches the frame.
    desc.maxUploadBytesPerFrame = 2ull << 20u;
    desc.unusedFrameLimit = 30;
    TextureStreamer streamer;
    if (!streamer.Initialize(desc, allocator).IsOk())
    {
        std::printf("streamer initialization failed\n");
        return 1;
    }
    PoolSink sink(desc.budgetBytes, desc.pageBytes);

    std::vector<PanelSource*> sources;
    std::vector<TextureHandle> handles;
    std::vector<Panel> panels;
    sources.reserve(kPanelCount);
    for (u32 index = 0; index < kPanelCount; ++index)
    {
        sources.push_back(new PanelSource(index));
        char name[32];
        std::snprintf(name, sizeof(name), "panel_%03u", index);
        auto handle = streamer.Register(*sources.back(), name);
        if (!handle.IsOk())
        {
            std::printf("register failed\n");
            return 1;
        }
        handles.push_back(handle.Value());
        panels.push_back(Panel{(index & 1u) != 0 ? kCorridorHalfWidth : -kCorridorHalfWidth,
                               static_cast<f32>(index / 2u) * kPanelSpacing});
    }

    // Level load: the tails only, so every panel can be drawn on frame one.
    streamer.Flush(sink);
    const StreamingStats loaded = streamer.Stats();
    std::printf("texture streaming: %u textures, %.1f MiB of texture data, budget %.1f MiB (%.2fx "
                "oversubscribed)\n",
                loaded.textures, static_cast<f64>(loaded.fullChainBytes) / 1048576.0,
                static_cast<f64>(desc.budgetBytes) / 1048576.0,
                static_cast<f64>(loaded.fullChainBytes) / static_cast<f64>(desc.budgetBytes));
    std::printf(
        "after level load: tails resident %.2f MiB in %u pages, %u textures without a tail\n",
        static_cast<f64>(loaded.tailBytes) / 1048576.0, loaded.usedPages,
        loaded.texturesWithoutTail);

    const f32 focal = static_cast<f32>(kScreenHeight) / (2.0f * std::tan(kVerticalFov * 0.5f));
    const f32 screenArea = static_cast<f32>(kScreenWidth) * static_cast<f32>(kScreenHeight);
    struct FrameRecord
    {
        f64 ms = 0.0;
        u32 frame = 0;
        u32 uploads = 0;
        u64 uploadBytes = 0;
        u32 requests = 0;
        u32 evictions = 0;
    };
    std::vector<f64> updateMs;
    std::vector<FrameRecord> records;
    updateMs.reserve(frames);
    records.reserve(frames);
    u64 missingSum = 0;
    u32 maxMissing = 0;
    u32 framesWithoutTail = 0;
    u32 maxUsedPages = 0;
    u32 hitches = 0;
    const u32 snapshotFrames[3] = {frames / 4, frames / 2, frames - 1};
    std::vector<Byte> view(800u * 450u * 4u);
    cyber::memory::DynamicArray<TextureResidency> snapshot(allocator);

    auto next = std::chrono::steady_clock::now();
    for (u32 frame = 0; frame < frames; ++frame)
    {
        const f32 cameraZ =
            static_cast<f32>(frame) * static_cast<f32>(kFrameSeconds) * kCameraSpeed - 10.0f;
        for (u32 index = 0; index < kPanelCount; ++index)
        {
            const f32 dz = panels[index].z - cameraZ;
            const f32 dx = panels[index].x;
            if (dz <= 0.3f || dz > kFarPlane)
            {
                continue;  // behind the camera or past the far plane: not drawn
            }
            const f32 distance = std::sqrt(dx * dx + dz * dz);
            // Panels face the corridor axis, so their projected width shrinks
            // with the grazing angle; height does not.
            const f32 pixels = kPanelSize * focal / distance;
            if (std::fabs(dx * focal / dz) > static_cast<f32>(kScreenWidth) * 0.5f + pixels)
            {
                continue;  // outside the horizontal field of view
            }
            const f32 coverage = std::min(
                pixels * pixels * (dx != 0.0f ? std::fabs(dz) / distance : 1.0f) / screenArea,
                1.0f);
            streamer.ReportUsage(handles[index],
                                 ComputeWantedMip(kTextureSize, kTextureSize, pixels), coverage);
        }

        const auto start = std::chrono::steady_clock::now();
        const StreamingStats stats = streamer.Update(sink);
        const f64 ms =
            std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        updateMs.push_back(ms);
        records.push_back(FrameRecord{ms, frame, stats.uploadsThisFrame, stats.uploadBytesThisFrame,
                                      stats.requestsThisFrame, stats.evictionsThisFrame});
        hitches += ms > kHitchMilliseconds ? 1u : 0u;
        missingSum += stats.missingLevels;
        maxMissing = std::max(maxMissing, stats.missingLevels);
        framesWithoutTail += stats.texturesWithoutTail > 0 ? 1u : 0u;
        maxUsedPages = std::max(maxUsedPages, stats.usedPages);

        for (const u32 shot : snapshotFrames)
        {
            if (frame == shot && streamer.Snapshot(snapshot))
            {
                // Only the 96 panels around the camera fit legibly; show those.
                const u32 first =
                    std::min(static_cast<u32>(std::max(cameraZ, 0.0f) / kPanelSpacing) * 2u,
                             kPanelCount - 96u);
                (void)RenderResidencyView(Span<const TextureResidency>(snapshot.Data() + first, 96),
                                          stats, 800, 450, Span<Byte>(view.data(), view.size()));
                char path[64];
                std::snprintf(path, sizeof(path), "texture_streaming_frame%04u.ppm", frame);
                if (!WritePpm(prefix + path, view, 800, 450))
                {
                    std::printf("could not write %s%s\n", prefix.c_str(), path);
                }
            }
        }

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
    const StreamingStats& summary = streamer.Stats();
    const f64 p99 = sorted[static_cast<usize>(static_cast<f64>(sorted.size() - 1) * 0.99)];
    std::printf("frames %u: streaming update mean %.3f ms, p99 %.3f ms, max %.3f ms; %u hitches "
                "over %.1f ms\n",
                frames, total / static_cast<f64>(sorted.size()), p99, sorted.back(), hitches,
                kHitchMilliseconds);
    std::printf("budget: peak %u of %u pages (%.1f%%), %llu uploads (%.1f MiB), %llu evictions\n",
                maxUsedPages, summary.totalPages,
                100.0 * maxUsedPages / std::max(summary.totalPages, 1u),
                static_cast<unsigned long long>(summary.totalUploads),
                static_cast<f64>(summary.totalUploadBytes) / 1048576.0,
                static_cast<unsigned long long>(summary.totalEvictions));
    std::printf("view: mean %.2f missing levels across visible panels, worst %u; %u frames had a "
                "texture without its tail\n",
                static_cast<f64>(missingSum) / frames, maxMissing, framesWithoutTail);

    std::sort(records.begin(), records.end(),
              [](const FrameRecord& lhs, const FrameRecord& rhs) { return lhs.ms > rhs.ms; });
    std::printf("slowest updates:\n");
    for (usize i = 0; i < records.size() && i < 5; ++i)
    {
        const FrameRecord& record = records[i];
        std::printf("  frame %4u: %.3f ms, %u uploads (%.2f MiB), %u requests, %u evictions\n",
                    record.frame, record.ms, record.uploads,
                    static_cast<f64>(record.uploadBytes) / 1048576.0, record.requests,
                    record.evictions);
    }

    const bool pass = hitches == 0 && maxUsedPages <= summary.totalPages &&
                      framesWithoutTail == 0 && !sink.overflow;
    std::printf("%s\n", pass ? "PASS: oversubscribed scene streamed within budget without hitches"
                             : "FAIL: see the numbers above");

    for (const TextureHandle handle : handles)
    {
        streamer.Unregister(handle, sink);
    }
    streamer.Shutdown();
    for (PanelSource* source : sources)
    {
        delete source;
    }
    return pass ? 0 : 1;
}
