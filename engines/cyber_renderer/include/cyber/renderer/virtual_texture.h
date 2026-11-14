#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/span.h>

namespace cyber::renderer
{

using cyber::core::Byte;
using cyber::core::f32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::usize;

/// Virtual texturing (Phase 33): one texture far larger than video memory --
/// a terrain, a world map, a megatexture -- cut into fixed-size pages, of which
/// only those the view samples are resident in a physical page cache.
///
/// - The virtual image has a mip pyramid; each mip is a grid of pages of
///   `pageSize` texels plus a `border` on every side, so bilinear (and up to
///   2x anisotropic) filtering inside a page never reads a neighbour's texels.
/// - The page table holds, for every page of every mip, where to sample it:
///   the finest resident page covering it -- itself, or an ancestor. A sample
///   therefore always resolves, sharp where the page is resident and blurrier
///   where it is still loading; the single top-mip page is pinned so the
///   fallback always exists.
/// - Feedback: whatever samples the texture reports the pages it wanted
///   (`RequestPage` / `RequestSample`, safe from any thread, e.g. a pixel
///   shader running on job threads). `Update` turns the frame's requests into
///   loads, coarsest first, produced on background threads, and copies at
///   most `maxUploadsPerFrame` into the cache, evicting the least recently
///   used pages the frame did not touch.
/// - GPU-ready data: the page table as one 32-bit entry per page per mip
///   (`EncodePageTableEntry`), the cache as an RGBA8 atlas, and per-frame
///   dirty lists of what changed, which a renderer uploads as an R32_UINT
///   page-table texture and an atlas texture. `Sample` performs the same
///   lookup on the CPU for the software renderer and for tests.

/// Fills one page: `(pageSize + 2 * border)^2` RGBA8 texels, row-major, the
/// border included, for page (pageX, pageY) of `mip`. Runs on IO threads,
/// concurrently for different pages; must be thread-safe. Returns false on
/// failure, and the page is requested again by later feedback.
class VirtualPageProducer
{
public:
    virtual ~VirtualPageProducer() = default;
    [[nodiscard]] virtual bool ProducePage(u32 mip, u32 pageX, u32 pageY,
                                           memory::Span<Byte> rgba) noexcept = 0;
};

struct VirtualTextureDesc
{
    /// Virtual extent in texels: powers of two, each at least `pageSize`.
    u32 width = 0;
    u32 height = 0;
    /// Payload texels per page side, a power of two.
    u32 pageSize = 128;
    /// Filtering border texels on each side of a page.
    u32 border = 4;
    /// Physical cache size in pages. The cache holds every pinned top-level
    /// page plus whatever the view needs.
    u32 cachePagesX = 32;
    u32 cachePagesY = 32;
    u32 ioThreads = 2;
    /// Pages being produced or waiting for upload at once.
    u32 maxInFlight = 32;
    /// Pages copied into the cache per Update: the per-frame main-thread cost.
    u32 maxUploadsPerFrame = 16;
};

struct VirtualTextureStats
{
    u32 mipCount = 0;
    u64 virtualPages = 0;
    u32 cachePages = 0;
    u32 residentPages = 0;
    u32 requestedThisFrame = 0;  // distinct pages the frame's feedback asked for
    u32 missingThisFrame = 0;    // of those, not resident at Update time
    u32 uploadsThisFrame = 0;
    u32 evictionsThisFrame = 0;
    u32 inFlight = 0;
    u32 producerErrorsThisFrame = 0;
    /// Requests deferred because every cache page was used this frame.
    u32 cacheFullThisFrame = 0;
    u64 totalUploads = 0;
    u64 totalEvictions = 0;
};

/// A resolved page-table entry: the cache page to sample and the mip it holds.
struct VirtualPageEntry
{
    u32 cacheX = 0;
    u32 cacheY = 0;
    u32 mip = 0;
};

/// The 32-bit GPU encoding of a page-table entry: cache x in bits 0-9, cache y
/// in 10-19, the resident mip in 20-24, bit 31 set for a valid entry.
[[nodiscard]] constexpr u32 EncodePageTableEntry(const VirtualPageEntry& entry) noexcept
{
    return (entry.cacheX & 0x3FFu) | ((entry.cacheY & 0x3FFu) << 10u) |
           ((entry.mip & 0x1Fu) << 20u) | 0x80000000u;
}

[[nodiscard]] constexpr VirtualPageEntry DecodePageTableEntry(u32 value) noexcept
{
    return VirtualPageEntry{value & 0x3FFu, (value >> 10u) & 0x3FFu, (value >> 20u) & 0x1Fu};
}

class VirtualTexture
{
public:
    VirtualTexture() noexcept = default;
    ~VirtualTexture();
    VirtualTexture(const VirtualTexture&) = delete;
    VirtualTexture& operator=(const VirtualTexture&) = delete;

    /// Allocates the cache and page tables, produces the pinned top-level
    /// page on the calling thread, and starts the IO threads. `producer` must
    /// outlive the texture.
    [[nodiscard]] core::Status<core::Error> Initialize(const VirtualTextureDesc& desc,
                                                       VirtualPageProducer& producer,
                                                       memory::Allocator& allocator) noexcept;
    void Shutdown() noexcept;

    [[nodiscard]] u32 MipCount() const noexcept;
    [[nodiscard]] u32 PagesX(u32 mip) const noexcept;
    [[nodiscard]] u32 PagesY(u32 mip) const noexcept;

    /// Feedback. Thread-safe and wait-free; duplicates within a frame cost a
    /// relaxed load.
    void RequestPage(u32 mip, u32 pageX, u32 pageY) noexcept;
    /// The page a sample at (u, v) in [0, 1) with level of detail `lod` needs.
    void RequestSample(f32 u, f32 v, f32 lod) noexcept;

    /// Main thread, once per frame, never concurrently with `Sample`.
    VirtualTextureStats Update() noexcept;
    /// Blocks until every requested page is resident or the cache is full of
    /// pages the requests need (for tests and loads).
    void Flush() noexcept;

    /// Where a sample at (u, v, lod) reads from right now.
    [[nodiscard]] VirtualPageEntry Resolve(f32 u, f32 v, f32 lod) const noexcept;
    /// Bilinear sample through the page table into `rgba` (0..1). Safe from
    /// many threads at once between Updates.
    void Sample(f32 u, f32 v, f32 lod, f32 rgba[4]) const noexcept;

    /// GPU data. The page table of `mip`, `PagesX(mip) * PagesY(mip)` entries
    /// in `EncodePageTableEntry` form; the atlas, RGBA8, `AtlasWidth()` texels
    /// wide; and what changed in the last Update.
    [[nodiscard]] memory::Span<const u32> PageTable(u32 mip) const noexcept;
    [[nodiscard]] memory::Span<const Byte> Atlas() const noexcept;
    [[nodiscard]] u32 AtlasWidth() const noexcept;
    [[nodiscard]] u32 AtlasHeight() const noexcept;
    /// Cache pages written by the last Update, as `y * cachePagesX + x`.
    [[nodiscard]] memory::Span<const u32> DirtyCachePages() const noexcept;
    /// True when the last Update changed any entry of `mip`'s page table.
    [[nodiscard]] bool PageTableDirty(u32 mip) const noexcept;

    [[nodiscard]] const VirtualTextureStats& Stats() const noexcept;

private:
    struct Impl;
    Impl* m_impl = nullptr;
};

}  // namespace cyber::renderer
