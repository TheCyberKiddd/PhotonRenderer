#include <cyber/memory/heap_allocator.h>
#include <cyber/renderer/texture_streaming.h>
#include <cyber/test/test.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <set>
#include <thread>
#include <vector>

namespace
{

using cyber::core::Byte;
using cyber::core::f32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::usize;
using cyber::memory::DynamicArray;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using namespace cyber::renderer;

constexpr u32 kPage = 1024;

// A square RGBA8-sized texture whose level bytes follow a recognisable
// pattern, so the sink can check every upload carries the right level.
class PatternSource final : public TextureMipSource
{
public:
    PatternSource(u32 id, u32 size) noexcept : m_id(id)
    {
        while (size > 0 && m_levels < kMaxStreamedLevels)
        {
            m_size[m_levels++] = size;
            size /= 2;
        }
    }
    u32 LevelCount() const noexcept override
    {
        return m_levels;
    }
    StreamingLevelInfo Level(u32 level) const noexcept override
    {
        const u32 size = m_size[level];
        return StreamingLevelInfo{size, size, static_cast<u64>(size) * size * 4u};
    }
    bool ReadLevel(u32 level, Span<Byte> destination) noexcept override
    {
        reads.fetch_add(1);
        if (failNextReadOfLevel.load() == static_cast<int>(level))
        {
            failNextReadOfLevel.store(-1);
            return false;
        }
        std::memset(destination.Data(), static_cast<int>(Expected(level)), destination.Size());
        return true;
    }
    [[nodiscard]] Byte Expected(u32 level) const noexcept
    {
        return static_cast<Byte>((m_id * 16u + level) & 0xFFu);
    }

    std::atomic<int> failNextReadOfLevel{-1};
    std::atomic<u32> reads{0};

private:
    u32 m_id = 0;
    u32 m_size[kMaxStreamedLevels] = {};
    u32 m_levels = 0;
};

// Models a GPU texture pool: tracks which pages are mapped to which level and
// checks the streamer never maps a page twice or uploads the wrong bytes.
class CheckingSink final : public TextureUploadSink
{
public:
    explicit CheckingSink(std::vector<PatternSource*> sources) : m_sources(std::move(sources)) {}

    // Points a handle index at the source now registered there.
    void SetSource(u32 index, PatternSource* source)
    {
        if (m_sources.size() <= index)
        {
            m_sources.resize(index + 1u, nullptr);
        }
        m_sources[index] = source;
    }

    void UploadLevels(TextureHandle texture, u32 firstLevel, u32 levelCount, Span<const Byte> data,
                      Span<const u32> pages) noexcept override
    {
        ++uploads;
        uploadOrder.push_back({texture.index, firstLevel});
        u64 offset = 0;
        PatternSource* source = m_sources[texture.index];
        for (u32 level = firstLevel; level < firstLevel + levelCount; ++level)
        {
            const u64 bytes = source->Level(level).bytes;
            for (u64 i = 0; i < bytes; i += 97)
            {
                if (data[static_cast<usize>(offset + i)] != source->Expected(level))
                {
                    wrongBytes = true;
                }
            }
            offset += bytes;
        }
        if (offset != data.Size() || pages.Size() * kPage < data.Size())
        {
            wrongSize = true;
        }
        for (usize i = 0; i < pages.Size(); ++i)
        {
            if (!mapped.insert(pages[i]).second)
            {
                doubleMapped = true;
            }
        }
        frameUploadBytes += data.Size();
    }

    void EvictLevel(TextureHandle, u32, Span<const u32> pages) noexcept override
    {
        ++evictions;
        for (usize i = 0; i < pages.Size(); ++i)
        {
            if (mapped.erase(pages[i]) != 1)
            {
                unmappedEviction = true;
            }
        }
    }

    u32 uploads = 0;
    u32 evictions = 0;
    u64 frameUploadBytes = 0;
    bool wrongBytes = false;
    bool wrongSize = false;
    bool doubleMapped = false;
    bool unmappedEviction = false;
    std::set<u32> mapped;
    std::vector<std::pair<u32, u32>> uploadOrder;

private:
    std::vector<PatternSource*> m_sources;
};

TextureStreamerDesc SmallDesc(u64 budget)
{
    TextureStreamerDesc desc;
    desc.budgetBytes = budget;
    desc.pageBytes = kPage;
    desc.ioThreads = 2;
    desc.maxInFlight = 8;
    desc.maxUploadBytesPerFrame = 1u << 30u;
    desc.unusedFrameLimit = 4;
    return desc;
}

// Runs frames until nothing is in flight, reporting `wanted` for every texture.
void Settle(TextureStreamer& streamer, CheckingSink& sink,
            const std::vector<TextureHandle>& handles, const std::vector<f32>& wanted,
            const std::vector<f32>& coverage, u32 frames = 400)
{
    for (u32 frame = 0; frame < frames; ++frame)
    {
        for (usize i = 0; i < handles.size(); ++i)
        {
            streamer.ReportUsage(handles[i], wanted[i], coverage[i]);
        }
        const StreamingStats stats = streamer.Update(sink);
        if (stats.inFlight == 0 && stats.requestsThisFrame == 0 && frame > 2)
        {
            return;
        }
        if (stats.inFlight != 0)
        {
            std::this_thread::yield();
        }
    }
}

}  // namespace

CG_TEST(wanted_mip_follows_texel_density)
{
    CG_EXPECT_NEAR(ComputeWantedMip(1024, 1024, 1024.0f), 0.0f, 1.0e-6f);
    CG_EXPECT_NEAR(ComputeWantedMip(1024, 512, 256.0f), 2.0f, 1.0e-5f);
    CG_EXPECT_NEAR(ComputeWantedMip(1024, 1024, 4096.0f), 0.0f, 1.0e-6f);
    // Tiling four times needs four times the texels.
    CG_EXPECT_NEAR(ComputeWantedMip(256, 256, 256.0f, 4.0f), 2.0f, 1.0e-5f);
    CG_EXPECT_TRUE(ComputeWantedMip(256, 256, 0.0f) >= static_cast<f32>(kMaxStreamedLevels));
}

CG_TEST(tail_loads_first_as_one_request_and_levels_stream_coarse_to_fine)
{
    HeapAllocator allocator;
    PatternSource source(1, 128);  // 128..1: levels 0-3 >= 1 KiB, 4.. are the tail
    CheckingSink sink({&source});
    TextureStreamer streamer;
    CG_REQUIRE(streamer.Initialize(SmallDesc(1u << 20u), allocator).IsOk());
    auto handle = streamer.Register(source, "pattern");
    CG_REQUIRE(handle.IsOk());

    streamer.Flush(sink);
    DynamicArray<TextureResidency> snapshot(allocator);
    CG_REQUIRE(streamer.Snapshot(snapshot));
    CG_REQUIRE(snapshot.Size() == 1u);
    const u32 tail = snapshot[0].tailLevel;
    CG_EXPECT_EQ(tail, 4u);  // 16x16x4 = 1 KiB is not smaller than a page
    CG_EXPECT_EQ(streamer.ResidentLevel(handle.Value()), tail);
    CG_EXPECT_EQ(sink.uploads, 1u);

    Settle(streamer, sink, {handle.Value()}, {0.0f}, {0.5f});
    CG_EXPECT_EQ(streamer.ResidentLevel(handle.Value()), 0u);
    CG_REQUIRE(sink.uploadOrder.size() == 5u);
    for (usize i = 1; i < sink.uploadOrder.size(); ++i)
    {
        CG_EXPECT_EQ(sink.uploadOrder[i].second, tail - static_cast<u32>(i));
    }
    CG_EXPECT_FALSE(sink.wrongBytes);
    CG_EXPECT_FALSE(sink.wrongSize);
    CG_EXPECT_FALSE(sink.doubleMapped);
}

CG_TEST(budget_holds_and_important_textures_win)
{
    HeapAllocator allocator;
    // Each 64x64 chain is about 21.3 KiB in 1 KiB pages; four want level 0
    // but the budget holds roughly two and a half.
    std::vector<std::unique_ptr<PatternSource>> sources;
    sources.reserve(4);
    for (u32 i = 0; i < 4; ++i)
    {
        sources.push_back(std::make_unique<PatternSource>(i, 64));
    }
    std::vector<PatternSource*> pointers;
    for (auto& source : sources)
    {
        pointers.push_back(source.get());
    }
    CheckingSink sink(pointers);
    TextureStreamer streamer;
    CG_REQUIRE(streamer.Initialize(SmallDesc(56u * kPage), allocator).IsOk());
    std::vector<TextureHandle> handles;
    for (auto& source : sources)
    {
        handles.push_back(streamer.Register(*source, "t").Value());
    }
    Settle(streamer, sink, handles, {0.0f, 0.0f, 0.0f, 0.0f}, {0.40f, 0.30f, 0.02f, 0.01f});

    const StreamingStats& stats = streamer.Stats();
    CG_EXPECT_TRUE(stats.usedPages <= stats.totalPages);
    CG_EXPECT_TRUE(stats.residentBytes <= stats.budgetBytes);
    CG_EXPECT_EQ(stats.texturesWithoutTail, 0u);
    CG_EXPECT_EQ(streamer.ResidentLevel(handles[0]), 0u);
    CG_EXPECT_EQ(streamer.ResidentLevel(handles[1]), 0u);
    CG_EXPECT_TRUE(streamer.ResidentLevel(handles[3]) > 0u);
    CG_EXPECT_TRUE(stats.missingLevels > 0u);
    CG_EXPECT_FALSE(sink.doubleMapped);
    CG_EXPECT_FALSE(sink.unmappedEviction);
    CG_EXPECT_EQ(sink.mapped.size(), static_cast<usize>(stats.usedPages));
    // Settled, not thrashing: a handful of evictions to seat the important
    // textures, then none. Partial evictions that free memory a blocked
    // request cannot use would show up here as an endless churn.
    CG_EXPECT_EQ(stats.inFlight, 0u);
    CG_EXPECT_TRUE(sink.evictions <= 8u);
}

CG_TEST(surplus_levels_are_evicted_before_needed_ones)
{
    HeapAllocator allocator;
    PatternSource a(0, 64);
    PatternSource b(1, 64);
    PatternSource c(2, 64);
    CheckingSink sink({&a, &b, &c});
    TextureStreamer streamer;
    CG_REQUIRE(streamer.Initialize(SmallDesc(48u * kPage), allocator).IsOk());
    const TextureHandle ha = streamer.Register(a, "a").Value();
    const TextureHandle hb = streamer.Register(b, "b").Value();
    const TextureHandle hc = streamer.Register(c, "c").Value();

    // A and B fill most of the budget at full resolution.
    Settle(streamer, sink, {ha, hb, hc}, {0.0f, 0.0f, 9.0f}, {0.1f, 0.1f, 0.0f});
    CG_REQUIRE(streamer.ResidentLevel(ha) == 0u);
    CG_REQUIRE(streamer.ResidentLevel(hb) == 0u);

    // A recedes (now needs level 2: levels 0-1 are surplus); C, less
    // important than B, appears at full resolution. C must be fed from A's
    // surplus, not from B.
    Settle(streamer, sink, {ha, hb, hc}, {2.0f, 0.0f, 0.0f}, {0.1f, 0.1f, 0.05f});
    CG_EXPECT_EQ(streamer.ResidentLevel(hb), 0u);
    CG_EXPECT_EQ(streamer.ResidentLevel(hc), 0u);
    CG_EXPECT_TRUE(streamer.ResidentLevel(ha) >= 1u);
}

CG_TEST(equally_important_textures_do_not_trade_memory)
{
    HeapAllocator allocator;
    PatternSource a(0, 64);
    PatternSource b(1, 64);
    CheckingSink sink({&a, &b});
    TextureStreamer streamer;
    // Room for one full chain plus part of the other.
    CG_REQUIRE(streamer.Initialize(SmallDesc(30u * kPage), allocator).IsOk());
    const TextureHandle ha = streamer.Register(a, "a").Value();
    const TextureHandle hb = streamer.Register(b, "b").Value();
    for (u32 frame = 0; frame < 300; ++frame)
    {
        streamer.ReportUsage(ha, 0.0f, 0.2f);
        streamer.ReportUsage(hb, 0.0f, 0.2f);
        (void)streamer.Update(sink);
        std::this_thread::yield();
    }
    streamer.Flush(sink);
    const u32 evictionsAfterSettling = sink.evictions;
    for (u32 frame = 0; frame < 200; ++frame)
    {
        streamer.ReportUsage(ha, 0.0f, 0.2f);
        streamer.ReportUsage(hb, 0.0f, 0.2f);
        (void)streamer.Update(sink);
    }
    streamer.Flush(sink);
    CG_EXPECT_EQ(sink.evictions, evictionsAfterSettling);
    CG_EXPECT_TRUE(streamer.Stats().budgetBlockedThisFrame <= 1u);
}

CG_TEST(unused_textures_give_way)
{
    HeapAllocator allocator;
    PatternSource a(0, 64);
    PatternSource b(1, 64);
    CheckingSink sink({&a, &b});
    TextureStreamer streamer;
    CG_REQUIRE(streamer.Initialize(SmallDesc(30u * kPage), allocator).IsOk());
    const TextureHandle ha = streamer.Register(a, "a").Value();
    const TextureHandle hb = streamer.Register(b, "b").Value();
    Settle(streamer, sink, {ha, hb}, {0.0f, 9.0f}, {0.5f, 0.0f});
    CG_REQUIRE(streamer.ResidentLevel(ha) == 0u);
    // A leaves the view entirely; B now needs everything.
    for (u32 frame = 0; frame < 400 && streamer.ResidentLevel(hb) != 0u; ++frame)
    {
        streamer.ReportUsage(hb, 0.0f, 0.5f);
        (void)streamer.Update(sink);
        std::this_thread::yield();
    }
    CG_EXPECT_EQ(streamer.ResidentLevel(hb), 0u);
    CG_EXPECT_TRUE(streamer.ResidentLevel(ha) > 0u);
}

CG_TEST(per_frame_upload_bytes_are_bounded)
{
    HeapAllocator allocator;
    std::vector<std::unique_ptr<PatternSource>> sources;
    sources.reserve(6);
    std::vector<PatternSource*> pointers;
    for (u32 i = 0; i < 6; ++i)
    {
        sources.push_back(std::make_unique<PatternSource>(i, 64));
    }
    for (auto& source : sources)
    {
        pointers.push_back(source.get());
    }
    CheckingSink sink(pointers);
    TextureStreamerDesc desc = SmallDesc(1u << 20u);
    desc.maxUploadBytesPerFrame = 8u * kPage;
    TextureStreamer streamer;
    CG_REQUIRE(streamer.Initialize(desc, allocator).IsOk());
    std::vector<TextureHandle> handles;
    for (auto& source : sources)
    {
        handles.push_back(streamer.Register(*source, "t").Value());
    }
    u64 largestRequest = 64u * 64u * 4u;
    for (u32 frame = 0; frame < 2000; ++frame)
    {
        for (const TextureHandle handle : handles)
        {
            streamer.ReportUsage(handle, 0.0f, 0.1f);
        }
        const StreamingStats stats = streamer.Update(sink);
        // One request may exceed the budget on its own; never more.
        CG_EXPECT_TRUE(
            stats.uploadBytesThisFrame <= desc.maxUploadBytesPerFrame ||
            (stats.uploadsThisFrame == 1u && stats.uploadBytesThisFrame <= largestRequest));
        if (stats.missingLevels == 0 && stats.inFlight == 0 && frame > 4)
        {
            break;
        }
        std::this_thread::yield();
    }
    for (const TextureHandle handle : handles)
    {
        CG_EXPECT_EQ(streamer.ResidentLevel(handle), 0u);
    }
}

CG_TEST(failed_reads_are_retried)
{
    HeapAllocator allocator;
    PatternSource source(3, 64);
    source.failNextReadOfLevel.store(1);
    CheckingSink sink({&source});
    TextureStreamer streamer;
    CG_REQUIRE(streamer.Initialize(SmallDesc(1u << 20u), allocator).IsOk());
    const TextureHandle handle = streamer.Register(source, "flaky").Value();
    u32 errors = 0;
    for (u32 frame = 0; frame < 400 && streamer.ResidentLevel(handle) != 0u; ++frame)
    {
        streamer.ReportUsage(handle, 0.0f, 0.5f);
        errors += streamer.Update(sink).ioErrorsThisFrame;
        std::this_thread::yield();
    }
    CG_EXPECT_EQ(errors, 1u);
    CG_EXPECT_EQ(streamer.ResidentLevel(handle), 0u);
    CG_EXPECT_FALSE(sink.wrongBytes);
}

CG_TEST(unregister_releases_everything_and_stales_the_handle)
{
    HeapAllocator allocator;
    PatternSource a(0, 64);
    PatternSource b(1, 64);
    CheckingSink sink({&a, &b});
    TextureStreamer streamer;
    CG_REQUIRE(streamer.Initialize(SmallDesc(1u << 20u), allocator).IsOk());
    const TextureHandle ha = streamer.Register(a, "a").Value();
    Settle(streamer, sink, {ha}, {0.0f}, {0.5f});
    CG_REQUIRE(!sink.mapped.empty());
    // Unregister while a read may be in flight for the other texture too.
    const TextureHandle hb = streamer.Register(b, "b").Value();
    streamer.ReportUsage(hb, 0.0f, 0.5f);
    (void)streamer.Update(sink);
    streamer.Unregister(ha, sink);
    streamer.Unregister(hb, sink);
    CG_EXPECT_FALSE(streamer.IsValid(ha));
    CG_EXPECT_EQ(streamer.ResidentLevel(ha), kNoLevel);
    (void)streamer.Update(sink);
    CG_EXPECT_TRUE(sink.mapped.empty());
    CG_EXPECT_EQ(streamer.Stats().usedPages, 0u);
    CG_EXPECT_FALSE(sink.unmappedEviction);

    // The slot is reused with a new generation; the old handle stays stale.
    PatternSource c(0, 32);
    const TextureHandle hc = streamer.Register(c, "c").Value();
    CG_EXPECT_EQ(hc.index, ha.index);
    CG_EXPECT_FALSE(hc == ha);
    CG_EXPECT_FALSE(streamer.IsValid(ha));
}

CG_TEST(residency_view_draws_the_budget_and_the_strips)
{
    HeapAllocator allocator;
    TextureResidency entry;
    entry.levelCount = 4;
    entry.tailLevel = 3;
    entry.residentLevel = 1;
    entry.wantedLevel = 0;
    entry.used = true;
    entry.residentBytes = 50;
    entry.fullChainBytes = 100;
    StreamingStats stats;
    stats.totalPages = 10;
    stats.usedPages = 5;
    const u32 width = 128;
    const u32 height = 64;
    std::vector<Byte> image(width * height * 4);
    const ResidencyViewColors colors;
    CG_REQUIRE(RenderResidencyView(Span<const TextureResidency>(&entry, 1), stats, width, height,
                                   Span<Byte>(image.data(), image.size()), colors));
    const auto pixel = [&](u32 x, u32 y)
    {
        u32 value = 0;
        std::memcpy(&value, image.data() + (static_cast<usize>(y) * width + x) * 4u, 4);
        return value;
    };
    CG_EXPECT_EQ(pixel(10, 6), colors.budgetUsed);
    CG_EXPECT_EQ(pixel(120, 6), colors.budgetFree);
    // Strip: level 0 missing (red), level 1-2 resident (green), tail teal.
    const u32 stripY = 30;
    const u32 box = (width - 2) / 4;
    CG_EXPECT_EQ(pixel(1 + box / 2, stripY), colors.missing);
    CG_EXPECT_EQ(pixel(1 + box + box / 2, stripY), colors.residentWanted);
    CG_EXPECT_EQ(pixel(1 + 3 * box + box / 2, stripY), colors.tail);
    CG_EXPECT_FALSE(RenderResidencyView(Span<const TextureResidency>(&entry, 1), stats, 8, 8,
                                        Span<Byte>(image.data(), image.size())));
}

// A long randomized run: textures registered and unregistered while their
// reads are in flight, wants and coverage changing every frame, reads failing
// at random, and a budget far below demand. After every frame: no page is
// mapped twice, the sink holds exactly the streamer's resident pages, every
// texture's resident chain is contiguous down to its tail, and the budget is
// never exceeded.
CG_TEST(randomized_churn_keeps_every_invariant)
{
    HeapAllocator allocator;
    constexpr u32 kSlots = 24;
    std::vector<std::unique_ptr<PatternSource>> sources;
    for (u32 i = 0; i < kSlots; ++i)
    {
        sources.push_back(std::make_unique<PatternSource>(i, i % 3 == 0 ? 128u : 64u));
    }
    CheckingSink sink({});
    TextureStreamerDesc desc = SmallDesc(160u * kPage);
    desc.maxUploadBytesPerFrame = 24u * kPage;
    TextureStreamer streamer;
    CG_REQUIRE(streamer.Initialize(desc, allocator).IsOk());

    std::vector<TextureHandle> handles(kSlots);
    std::vector<bool> live(kSlots, false);
    u32 seed = 0x2545F491u;
    const auto random = [&seed](u32 bound)
    {
        seed ^= seed << 13u;
        seed ^= seed >> 17u;
        seed ^= seed << 5u;
        return seed % bound;
    };
    u32 violations = 0;
    u32 registrations = 0;
    u32 unregistrations = 0;
    DynamicArray<TextureResidency> snapshot(allocator);
    for (u32 frame = 0; frame < 3000; ++frame)
    {
        // Churn: register or unregister a random slot now and then. Handles
        // are only reused once their slot's reads have drained, so a slot is
        // registered into the index the streamer picks.
        if (random(8) == 0)
        {
            const u32 slot = random(kSlots);
            if (live[slot])
            {
                streamer.Unregister(handles[slot], sink);
                live[slot] = false;
                ++unregistrations;
            }
        }
        if (random(6) == 0)
        {
            for (u32 slot = 0; slot < kSlots; ++slot)
            {
                if (!live[slot])
                {
                    auto handle = streamer.Register(*sources[slot], "churn");
                    CG_REQUIRE(handle.IsOk());
                    // The sink checks bytes by handle index; point it at the
                    // source the streamer put at this index.
                    sink.SetSource(handle.Value().index, sources[slot].get());
                    handles[slot] = handle.Value();
                    live[slot] = true;
                    ++registrations;
                    break;
                }
            }
        }
        if (random(40) == 0)
        {
            sources[random(kSlots)]->failNextReadOfLevel.store(static_cast<int>(random(4)));
        }
        for (u32 slot = 0; slot < kSlots; ++slot)
        {
            if (live[slot] && random(4) != 0)
            {
                streamer.ReportUsage(handles[slot], static_cast<f32>(random(5)),
                                     static_cast<f32>(random(100)) / 400.0f);
            }
        }
        const StreamingStats stats = streamer.Update(sink);
        violations += stats.usedPages <= stats.totalPages ? 0u : 1u;
        violations +=
            sink.mapped.size() == static_cast<usize>(stats.usedPages - stats.reservedPages) ? 0u
                                                                                            : 1u;
        CG_REQUIRE(streamer.Snapshot(snapshot));
        for (usize i = 0; i < snapshot.Size(); ++i)
        {
            const TextureResidency& entry = snapshot[i];
            // Resident means the whole chain from residentLevel to the tail.
            if (entry.residentLevel < entry.levelCount && entry.residentLevel > entry.tailLevel)
            {
                ++violations;
            }
            if (entry.loadingLevel != kNoLevel && entry.loadingLevel != entry.tailLevel &&
                entry.loadingLevel + 1 != entry.residentLevel)
            {
                ++violations;
            }
        }
        if (random(16) == 0)
        {
            std::this_thread::yield();
        }
    }
    for (u32 slot = 0; slot < kSlots; ++slot)
    {
        if (live[slot])
        {
            streamer.Unregister(handles[slot], sink);
        }
    }
    (void)streamer.Update(sink);
    CG_EXPECT_EQ(violations, 0u);
    CG_EXPECT_TRUE(registrations > 50u);
    CG_EXPECT_TRUE(unregistrations > 50u);
    CG_EXPECT_TRUE(sink.mapped.empty());
    CG_EXPECT_EQ(streamer.Stats().usedPages, 0u);
    CG_EXPECT_FALSE(sink.doubleMapped);
    CG_EXPECT_FALSE(sink.unmappedEviction);
    CG_EXPECT_FALSE(sink.wrongBytes);
    CG_EXPECT_FALSE(sink.wrongSize);
}
