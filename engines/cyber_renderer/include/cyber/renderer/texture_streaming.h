#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/span.h>

namespace cyber::renderer
{

using cyber::core::Byte;
using cyber::core::f32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::usize;

/// Texture mip streaming: a residency manager that keeps the mips the camera
/// needs inside a fixed video-memory budget, loading them on background IO
/// threads and evicting the least useful when space runs out.
///
/// The model follows Direct3D 12 reserved (tiled) resources:
///
/// - Memory is a pool of fixed-size pages (64 KiB, the D3D12 tile size). A mip
///   level occupies whole pages, which need not be contiguous, so the pool
///   never fragments.
/// - The coarsest levels, each smaller than a page, form the packed mip tail.
///   It is loaded first, as one request, and is never evicted: it is the
///   low-resolution placeholder every texture can always be drawn with.
/// - Finer levels stream one at a time per texture, coarsest first, so a
///   texture's resident levels are always a contiguous chain down to the
///   tail and the sampler clamp is a single number (`ResidentLevel`).
///
/// The renderer reports, per frame, how fine each texture needs to be
/// (`ReportUsage`, from `ComputeWantedMip`) and how much screen it covers. The
/// main-thread `Update` cost is bounded: it is linear in the texture count and
/// copies at most `maxUploadBytesPerFrame` into the sink, which is what keeps
/// streaming from causing frame hitches.

inline constexpr u32 kMaxStreamedLevels = 16;
inline constexpr u32 kNoLevel = 0xFFFFFFFFu;

struct TextureHandle
{
    u32 index = kNoLevel;
    u32 generation = 0;

    [[nodiscard]] constexpr bool IsValid() const noexcept
    {
        return index != kNoLevel;
    }
    friend constexpr bool operator==(TextureHandle lhs, TextureHandle rhs) noexcept
    {
        return lhs.index == rhs.index && lhs.generation == rhs.generation;
    }
};

struct StreamingLevelInfo
{
    u32 width = 0;
    u32 height = 0;
    u64 bytes = 0;
};

/// Where a texture's mip data comes from: a file, a package, or a procedural
/// generator. `ReadLevel` runs on IO threads and may block; it must be safe to
/// call concurrently for different levels of the same source. The other two
/// functions are called on the main thread only.
class TextureMipSource
{
public:
    virtual ~TextureMipSource() = default;
    /// Level 0 is the finest. At most kMaxStreamedLevels.
    [[nodiscard]] virtual u32 LevelCount() const noexcept = 0;
    [[nodiscard]] virtual StreamingLevelInfo Level(u32 level) const noexcept = 0;
    /// Fills `destination` (exactly Level(level).bytes long). Returns false on
    /// an IO error; the level is then retried on a later frame.
    [[nodiscard]] virtual bool ReadLevel(u32 level, memory::Span<Byte> destination) noexcept = 0;

    /// Pages a request for levels `firstLevel..firstLevel+levelCount-1`
    /// occupies. Zero, the default, derives it from the bytes. A source backed
    /// by a reserved GPU texture returns the driver's tile count instead, which
    /// can exceed the bytes (partial tiles at mip edges, or small mips a driver
    /// gives tiles of their own).
    [[nodiscard]] virtual u32 PageCount(u32 firstLevel, u32 levelCount) const noexcept
    {
        (void)firstLevel;
        (void)levelCount;
        return 0;
    }
    /// First level of the packed tail, or kNoLevel (the default) to derive it:
    /// the coarsest run of levels each smaller than a page. A GPU-backed
    /// source returns where the driver's packed mips begin.
    [[nodiscard]] virtual u32 TailLevel() const noexcept
    {
        return kNoLevel;
    }
};

/// Receives residency changes on the main thread, inside `Update`. A GPU
/// backend maps `pages` into the texture's reserved resource and copies
/// `data` through its upload heap; the software path copies into its own
/// storage. `data` is only valid for the duration of the call.
class TextureUploadSink
{
public:
    virtual ~TextureUploadSink() = default;
    /// `firstLevel..firstLevel+levelCount-1` arrive together (more than one
    /// only for the packed tail), laid out back to back in `data`.
    virtual void UploadLevels(TextureHandle texture, u32 firstLevel, u32 levelCount,
                              memory::Span<const Byte> data,
                              memory::Span<const u32> pages) noexcept = 0;
    virtual void EvictLevel(TextureHandle texture, u32 level,
                            memory::Span<const u32> pages) noexcept = 0;
};

struct TextureStreamerDesc
{
    /// Video memory the streamer may occupy, tails included. Integrated GPUs
    /// share system memory, so keep this conservative there.
    u64 budgetBytes = 256ull << 20u;
    /// Allocation granularity. 64 KiB matches D3D12 tiles.
    u32 pageBytes = 64u << 10u;
    /// Background reader threads.
    u32 ioThreads = 2;
    /// Requests that may be reading or waiting for upload at once.
    u32 maxInFlight = 16;
    /// Upload budget per Update. At least one ready request is always applied
    /// per frame, so a level larger than the budget still makes progress.
    u64 maxUploadBytesPerFrame = 8ull << 20u;
    /// Frames without a ReportUsage after which a texture only wants its tail
    /// and its finer levels become the first eviction candidates.
    u32 unusedFrameLimit = 60;
    /// Added to every wanted mip; positive values trade sharpness for memory.
    f32 mipBias = 0.0f;
};

/// Per-texture residency, for the debug view and for tests.
struct TextureResidency
{
    TextureHandle handle;
    const char* name = "";
    u32 levelCount = 0;
    /// First level of the packed tail.
    u32 tailLevel = 0;
    /// Finest level whose whole chain is resident; levelCount while the tail
    /// is still loading. Sample with MinLOD = residentLevel.
    u32 residentLevel = 0;
    /// Finest level the last ReportUsage asked for.
    u32 wantedLevel = 0;
    /// Level being read or waiting for upload, or kNoLevel.
    u32 loadingLevel = kNoLevel;
    u64 residentBytes = 0;
    u64 fullChainBytes = 0;
    f32 priority = 0.0f;
    bool used = false;
};

struct StreamingStats
{
    u64 budgetBytes = 0;
    u64 pageBytes = 0;
    u32 totalPages = 0;
    u32 usedPages = 0;
    u32 reservedPages = 0;  // allocated to in-flight requests
    u64 residentBytes = 0;  // payload bytes of resident levels
    u64 tailBytes = 0;
    u64 stagingBytes = 0;
    u64 fullChainBytes = 0;  // every level of every texture: the demand
    u32 textures = 0;
    u32 inFlight = 0;
    /// Sum over used textures of (residentLevel - wantedLevel) where positive:
    /// how many levels the view is still waiting for.
    u32 missingLevels = 0;
    /// Textures whose tail is not resident yet (drawn with the fallback).
    u32 texturesWithoutTail = 0;
    u32 uploadsThisFrame = 0;
    u64 uploadBytesThisFrame = 0;
    u32 evictionsThisFrame = 0;
    u32 requestsThisFrame = 0;
    u32 ioErrorsThisFrame = 0;
    u64 totalUploads = 0;
    u64 totalEvictions = 0;
    u64 totalUploadBytes = 0;
    /// Requests that had to wait because the budget was full of levels more
    /// important than theirs.
    u32 budgetBlockedThisFrame = 0;
};

/// The mip a surface needs: log2 of texels per screen pixel along the
/// surface's larger axis. `screenPixels` is the projected size of the surface
/// in pixels, `uvScale` how many times the texture repeats across it.
[[nodiscard]] f32 ComputeWantedMip(u32 textureWidth, u32 textureHeight, f32 screenPixels,
                                   f32 uvScale = 1.0f) noexcept;

class TextureStreamer
{
public:
    TextureStreamer() noexcept = default;
    ~TextureStreamer();
    TextureStreamer(const TextureStreamer&) = delete;
    TextureStreamer& operator=(const TextureStreamer&) = delete;
    TextureStreamer(TextureStreamer&&) = delete;
    TextureStreamer& operator=(TextureStreamer&&) = delete;

    /// Starts the IO threads. Fails on a zero budget, a budget smaller than one
    /// page, or thread creation failure.
    [[nodiscard]] core::Status<core::Error> Initialize(const TextureStreamerDesc& desc,
                                                       memory::Allocator& allocator) noexcept;
    /// Stops the IO threads after their current read; pending requests are
    /// dropped. Called by the destructor.
    void Shutdown() noexcept;

    /// The source must outlive Unregister. Its tail is requested immediately.
    [[nodiscard]] core::Result<TextureHandle, core::Error> Register(TextureMipSource& source,
                                                                    const char* name) noexcept;
    /// Waits for the texture's in-flight read, then releases its pages through
    /// `sink` (EvictLevel per resident level).
    void Unregister(TextureHandle texture, TextureUploadSink& sink) noexcept;

    /// Per-frame feedback. `wantedMip` from ComputeWantedMip (fractional mips
    /// round down: the finer level is wanted); `screenCoverage` is the
    /// texture's share of the frame (0..1, summed over its uses) and orders
    /// requests and evictions. The largest report of a frame wins.
    void ReportUsage(TextureHandle texture, f32 wantedMip, f32 screenCoverage) noexcept;

    /// Applies finished reads, schedules new ones, and evicts to make room.
    /// Call once per frame on the main thread.
    StreamingStats Update(TextureUploadSink& sink) noexcept;

    [[nodiscard]] u32 ResidentLevel(TextureHandle texture) const noexcept;
    [[nodiscard]] bool IsValid(TextureHandle texture) const noexcept;
    [[nodiscard]] const StreamingStats& Stats() const noexcept;
    /// Replaces `out` with one entry per registered texture, in handle order.
    [[nodiscard]] bool Snapshot(memory::DynamicArray<TextureResidency>& out) const noexcept;

    /// Blocks until no request is in flight and every finished read has been
    /// applied (repeatedly calling Update). For tests and level loads.
    void Flush(TextureUploadSink& sink) noexcept;

private:
    struct Impl;
    Impl* m_impl = nullptr;
};

/// Debug view colours, documented here so screenshots can be read. Values are
/// 0xAABBGGRR, i.e. R, G, B, A in memory order.
struct ResidencyViewColors
{
    u32 residentWanted = 0xFF3CB44Bu;   // green: resident and needed
    u32 residentSurplus = 0xFFE6A03Cu;  // blue: resident, finer than needed
    u32 loading = 0xFF19E1FFu;          // yellow: being read or uploaded
    u32 missing = 0xFF3030E6u;          // red: needed, not resident
    u32 absent = 0xFF3A3A3Au;           // grey: not needed, not resident
    u32 tail = 0xFFB4A046u;             // teal: packed tail, always resident
    u32 background = 0xFF141414u;
    u32 budgetUsed = 0xFF3CB44Bu;
    u32 budgetReserved = 0xFF19E1FFu;
    u32 budgetFree = 0xFF505050u;
};

/// Draws the residency view into a tightly packed RGBA8 image: a budget bar
/// across the top (resident, in flight, free), then one cell per texture with
/// a strip of boxes from level 0 (left) to one box for the packed tail
/// (right), coloured by
/// state, and a bar under the strip proportional to the texture's resident
/// share of its full chain. Returns false when the image is too small.
[[nodiscard]] bool RenderResidencyView(memory::Span<const TextureResidency> textures,
                                       const StreamingStats& stats, u32 width, u32 height,
                                       memory::Span<Byte> rgba,
                                       const ResidencyViewColors& colors = {}) noexcept;

}  // namespace cyber::renderer
