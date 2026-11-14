// Virtual texturing (Phase 33): the page table always resolves to correct
// data -- the page itself when resident, otherwise an ancestor -- through
// loads, LRU evictions and a cache too small for the request set; bilinear
// samples are exact inside pages and across page borders; feedback from many
// threads at once is counted once per page.

#include <cyber/memory/heap_allocator.h>
#include <cyber/renderer/virtual_texture.h>
#include <cyber/test/test.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

namespace
{

using cyber::core::Byte;
using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::core::u8;
using cyber::core::usize;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;
using namespace cyber::renderer;

constexpr u32 kSize = 4096;  // 32x32 pages of 128 at mip 0, 6 mips
constexpr u32 kPage = 128;
constexpr u32 kBorder = 4;

// Colour of virtual texel (x, y) of `mip`. Distinct per mip and smooth enough
// that a bilinear sample between two texels is their exact average in 8 bits
// when both differ by an even step.
void TexelColour(u32 mip, u32 x, u32 y, u8 out[4])
{
    out[0] = static_cast<u8>((x * 2u) & 0xFFu);
    out[1] = static_cast<u8>((y * 2u) & 0xFFu);
    out[2] = static_cast<u8>(mip * 40u);
    out[3] = 255;
}

class TestProducer final : public VirtualPageProducer
{
public:
    bool ProducePage(u32 mip, u32 pageX, u32 pageY, Span<Byte> rgba) noexcept override
    {
        produced.fetch_add(1);
        if (failures.load() > 0)
        {
            failures.fetch_sub(1);
            return false;
        }
        const u32 stride = kPage + 2u * kBorder;
        const i32 extent = static_cast<i32>(kSize >> mip);
        for (u32 j = 0; j < stride; ++j)
        {
            for (u32 i = 0; i < stride; ++i)
            {
                // Border texels are the neighbouring pages' texels, clamped at
                // the image edge, exactly as a cooker would bake them.
                const i32 gx = std::clamp(
                    static_cast<i32>(pageX * kPage + i) - static_cast<i32>(kBorder), 0, extent - 1);
                const i32 gy = std::clamp(
                    static_cast<i32>(pageY * kPage + j) - static_cast<i32>(kBorder), 0, extent - 1);
                u8 colour[4];
                TexelColour(mip, static_cast<u32>(gx), static_cast<u32>(gy), colour);
                for (u32 c = 0; c < 4; ++c)
                {
                    rgba[(static_cast<usize>(j) * stride + i) * 4u + c] =
                        static_cast<Byte>(colour[c]);
                }
            }
        }
        return true;
    }
    std::atomic<u32> produced{0};
    std::atomic<u32> failures{0};
};

VirtualTextureDesc Desc(u32 cacheSide)
{
    VirtualTextureDesc desc;
    desc.width = kSize;
    desc.height = kSize;
    desc.pageSize = kPage;
    desc.border = kBorder;
    desc.cachePagesX = cacheSide;
    desc.cachePagesY = cacheSide;
    desc.ioThreads = 2;
    desc.maxInFlight = 16;
    desc.maxUploadsPerFrame = 8;
    return desc;
}

// The sample at the centre of virtual texel (x, y) of `mip`, at that mip's lod.
bool CentreMatches(const VirtualTexture& vt, u32 mip, u32 x, u32 y)
{
    const f32 extent = static_cast<f32>(kSize >> mip);
    const f32 u = (static_cast<f32>(x) + 0.5f) / extent;
    const f32 v = (static_cast<f32>(y) + 0.5f) / extent;
    const VirtualPageEntry entry = vt.Resolve(u, v, static_cast<f32>(mip));
    // What should be there: the texel covering (u, v) in the resolved mip.
    const u32 shift = entry.mip - mip;
    u8 want[4];
    TexelColour(entry.mip, x >> shift, y >> shift, want);
    f32 got[4];
    vt.Sample(u, v, static_cast<f32>(mip), got);
    if (shift > 0)
    {
        // A coarser fallback is sampled between its texels; check the mip tag
        // only, which is constant across a mip.
        return std::fabs(got[2] * 255.0f - static_cast<f32>(want[2])) < 0.5f;
    }
    for (u32 c = 0; c < 4; ++c)
    {
        if (std::fabs(got[c] * 255.0f - static_cast<f32>(want[c])) > 0.5f)
        {
            return false;
        }
    }
    return true;
}

}  // namespace

CG_TEST(top_level_is_pinned_and_everything_resolves_to_it_at_start)
{
    HeapAllocator allocator;
    TestProducer producer;
    VirtualTexture vt;
    CG_REQUIRE(vt.Initialize(Desc(8), producer, allocator).IsOk());
    CG_EXPECT_EQ(vt.MipCount(), 6u);  // 4096 .. 128
    CG_EXPECT_EQ(vt.PagesX(0), 32u);
    CG_EXPECT_EQ(vt.Stats().residentPages, 1u);
    CG_EXPECT_EQ(vt.Resolve(0.7f, 0.2f, 0.0f).mip, 5u);
    CG_EXPECT_TRUE(CentreMatches(vt, 0, 3000, 1000));
    // Every mip's table is dirty at start so a renderer uploads it whole.
    for (u32 mip = 0; mip < vt.MipCount(); ++mip)
    {
        CG_EXPECT_TRUE(vt.PageTableDirty(mip));
        CG_EXPECT_EQ(vt.PageTable(mip).Size(), static_cast<usize>(vt.PagesX(mip)) * vt.PagesY(mip));
    }
}

CG_TEST(requested_pages_load_coarse_to_fine_and_sample_exactly)
{
    HeapAllocator allocator;
    TestProducer producer;
    VirtualTexture vt;
    CG_REQUIRE(vt.Initialize(Desc(16), producer, allocator).IsOk());
    vt.RequestSample(0.31f, 0.62f, 0.0f);
    vt.Flush();
    // The full chain from the top down to the requested page is resident.
    CG_EXPECT_EQ(vt.Resolve(0.31f, 0.62f, 0.0f).mip, 0u);
    for (u32 mip = 1; mip < vt.MipCount(); ++mip)
    {
        CG_EXPECT_EQ(vt.Resolve(0.31f, 0.62f, static_cast<f32>(mip)).mip, mip);
    }
    CG_EXPECT_EQ(vt.Stats().residentPages, 6u);
    // Exact at texel centres inside the page.
    const u32 x0 = static_cast<u32>(0.31f * kSize);
    const u32 y0 = static_cast<u32>(0.62f * kSize);
    u32 wrong = 0;
    for (u32 dy = 0; dy < 8; ++dy)
    {
        for (u32 dx = 0; dx < 8; ++dx)
        {
            wrong += CentreMatches(vt, 0, (x0 & ~(kPage - 1u)) + dx * 16u,
                                   (y0 & ~(kPage - 1u)) + dy * 16u)
                         ? 0u
                         : 1u;
        }
    }
    CG_EXPECT_EQ(wrong, 0u);

    // Bilinear across the page's right edge: halfway between the last texel
    // of this page and the first of the next, which the border supplies. The
    // two red values differ by 2, so the exact average is representable.
    const u32 edge = (x0 & ~(kPage - 1u)) + kPage - 1u;
    const f32 u = (static_cast<f32>(edge) + 1.0f) / static_cast<f32>(kSize);
    const f32 v = (static_cast<f32>(y0) + 0.5f) / static_cast<f32>(kSize);
    f32 got[4];
    vt.Sample(u, v, 0.0f, got);
    u8 left[4];
    u8 right[4];
    TexelColour(0, edge, y0, left);
    TexelColour(0, edge + 1u, y0, right);
    CG_EXPECT_NEAR(got[0] * 255.0f, 0.5f * (left[0] + right[0]), 0.51f);
}

CG_TEST(lru_eviction_keeps_every_page_table_entry_correct)
{
    HeapAllocator allocator;
    TestProducer producer;
    VirtualTexture vt;
    // A 6x6 cache (35 pages besides the pinned top) against a sweep across
    // every mip-0 page: constant eviction.
    CG_REQUIRE(vt.Initialize(Desc(6), producer, allocator).IsOk());
    u32 wrong = 0;
    for (u32 frame = 0; frame < 400; ++frame)
    {
        // A moving 3x3 window of mip-0 pages plus their coarser context.
        const u32 cx = (frame / 4u) % 30u;
        const u32 cy = (frame / 120u) % 30u;
        for (u32 dy = 0; dy < 3; ++dy)
        {
            for (u32 dx = 0; dx < 3; ++dx)
            {
                vt.RequestPage(0, cx + dx, cy + dy);
            }
        }
        vt.RequestPage(2, cx / 4u, cy / 4u);
        (void)vt.Update();
        // Every page of every mip resolves to the right data.
        if (frame % 25 == 0)
        {
            for (u32 mip = 0; mip < vt.MipCount(); ++mip)
            {
                for (u32 py = 0; py < vt.PagesY(mip); ++py)
                {
                    for (u32 px = 0; px < vt.PagesX(mip); ++px)
                    {
                        wrong +=
                            CentreMatches(vt, mip, px * kPage + 37u, py * kPage + 91u) ? 0u : 1u;
                    }
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CG_EXPECT_EQ(wrong, 0u);
    CG_EXPECT_TRUE(vt.Stats().totalEvictions > 0u);
    CG_EXPECT_TRUE(vt.Stats().residentPages <= 36u);
    // The top page is pinned: the coarsest lod still resolves to it.
    CG_EXPECT_EQ(vt.Resolve(0.5f, 0.5f, 5.0f).mip, 5u);
}

CG_TEST(a_request_set_larger_than_the_cache_degrades_gracefully)
{
    HeapAllocator allocator;
    TestProducer producer;
    VirtualTexture vt;
    CG_REQUIRE(vt.Initialize(Desc(4), producer, allocator).IsOk());
    u32 wrong = 0;
    u32 fullFrames = 0;
    for (u32 frame = 0; frame < 120; ++frame)
    {
        for (u32 y = 0; y < 8; ++y)
        {
            for (u32 x = 0; x < 8; ++x)
            {
                vt.RequestPage(0, x, y);
            }
        }
        fullFrames += vt.Update().cacheFullThisFrame > 0 ? 1u : 0u;
        // Frames take time, as in a game, so the IO threads deliver.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    for (u32 y = 0; y < 8; ++y)
    {
        for (u32 x = 0; x < 8; ++x)
        {
            wrong += CentreMatches(vt, 0, x * kPage + 5u, y * kPage + 5u) ? 0u : 1u;
        }
    }
    CG_EXPECT_EQ(wrong, 0u);
    // Under pressure either way: evicting to make room, or refusing an
    // upload because every cache page served this frame.
    CG_EXPECT_TRUE(fullFrames > 0u || vt.Stats().totalEvictions > 0u);
    CG_EXPECT_TRUE(vt.Stats().residentPages <= 16u);
}

CG_TEST(producer_failures_are_retried_by_later_feedback)
{
    HeapAllocator allocator;
    TestProducer producer;
    VirtualTexture vt;
    CG_REQUIRE(vt.Initialize(Desc(8), producer, allocator).IsOk());
    producer.failures.store(3);
    u32 errors = 0;
    for (u32 frame = 0; frame < 300 && vt.Resolve(0.9f, 0.1f, 0.0f).mip != 0u; ++frame)
    {
        vt.RequestSample(0.9f, 0.1f, 0.0f);
        errors += vt.Update().producerErrorsThisFrame;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CG_EXPECT_EQ(errors, 3u);
    CG_EXPECT_EQ(vt.Resolve(0.9f, 0.1f, 0.0f).mip, 0u);
}

CG_TEST(feedback_from_many_threads_counts_each_page_once)
{
    HeapAllocator allocator;
    TestProducer producer;
    VirtualTexture vt;
    CG_REQUIRE(vt.Initialize(Desc(8), producer, allocator).IsOk());
    std::vector<std::thread> workers;
    for (u32 t = 0; t < 4; ++t)
    {
        workers.emplace_back(
            [&vt, t]()
            {
                for (u32 i = 0; i < 20000; ++i)
                {
                    // 16 distinct mip-0 pages, hammered by every thread.
                    vt.RequestPage(0, (i + t) % 4u, ((i + t) / 4u) % 4u);
                }
            });
    }
    for (auto& worker : workers)
    {
        worker.join();
    }
    CG_EXPECT_EQ(vt.Update().requestedThisFrame, 16u);
}

CG_TEST(page_table_encoding_round_trips_and_dirty_lists_report_uploads)
{
    const VirtualPageEntry entry{1023, 517, 17};
    const VirtualPageEntry decoded = DecodePageTableEntry(EncodePageTableEntry(entry));
    CG_EXPECT_EQ(decoded.cacheX, 1023u);
    CG_EXPECT_EQ(decoded.cacheY, 517u);
    CG_EXPECT_EQ(decoded.mip, 17u);
    CG_EXPECT_TRUE((EncodePageTableEntry(entry) & 0x80000000u) != 0u);

    HeapAllocator allocator;
    TestProducer producer;
    VirtualTexture vt;
    CG_REQUIRE(vt.Initialize(Desc(8), producer, allocator).IsOk());
    (void)vt.Update();
    CG_EXPECT_TRUE(vt.DirtyCachePages().Empty());
    vt.RequestPage(4, 1, 1);
    vt.Flush();
    // The last Update that installed the page reported it; check the table
    // and the atlas agree with the GPU encoding.
    const Span<const cyber::core::u32> table = vt.PageTable(4);
    const VirtualPageEntry installed = DecodePageTableEntry(table[1 * vt.PagesX(4) + 1]);
    CG_EXPECT_EQ(installed.mip, 4u);
    CG_EXPECT_EQ(vt.AtlasWidth(), 8u * (kPage + 2u * kBorder));
    CG_EXPECT_EQ(vt.Atlas().Size(), static_cast<usize>(vt.AtlasWidth()) * vt.AtlasHeight() * 4u);
}
