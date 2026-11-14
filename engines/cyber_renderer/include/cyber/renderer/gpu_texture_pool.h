#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/renderer/texture_streaming.h>
#include <cyber/rhi/command_list.h>
#include <cyber/rhi/device.h>
#include <cyber/rhi/texture.h>

namespace cyber::renderer
{

/// The GPU side of texture streaming (Phase 33): a `TextureUploadSink` that
/// backs every streamed texture with an RHI reserved texture and the
/// streamer's page pool with one tile heap, page N being heap tile N.
///
/// - Registering a texture creates its reserved texture and wraps its source
///   in `TiledSource`, which tells the streamer the driver's tile counts and
///   where the driver's packed mips begin, so a page plan is a tile plan.
/// - An upload maps the level's tiles onto the pages it was given
///   (`UpdateTileMappings`), repacks the data into the per-frame upload buffer
///   at the API's 512-byte offsets and 256-byte row pitches, and records the
///   copy with the barriers around it.
/// - An eviction unmaps the level's tiles.
///
/// Ordering is what makes immediate page reuse safe: tile mapping updates run
/// on the same queue as the copies and the frame's draws, so a draw submitted
/// before an eviction finishes with the old mapping before the remap runs, and
/// a draw submitted after it clamps to the texture's new resident level.
///
/// Per frame: `BeginFrame`, `TextureStreamer::Update(pool)`, `EndFrame`, then
/// record draws that sample `Texture(handle)` with `MinLOD = ResidentLevel`.
struct GpuTexturePoolDesc
{
    /// Heap tiles; match `TextureStreamerDesc::budgetBytes / pageBytes`.
    u32 tileCount = 0;
    /// Upload buffers in rotation; a buffer is reused once the frame that
    /// filled it has finished on the GPU.
    u32 framesInFlight = 3;
    /// Bytes each frame's upload buffer holds. Match the streamer's
    /// `maxUploadBytesPerFrame` plus alignment slack; an upload that does not
    /// fit flushes the frame so far and waits for it.
    u64 uploadBytesPerFrame = 9ull << 20u;
    rhi::QueueKind queue = rhi::QueueKind::Graphics;
    const char* label = "texture_pool";
};

struct GpuTexturePoolStats
{
    u64 uploadedBytes = 0;
    u32 uploads = 0;
    u32 evictions = 0;
    u32 tileMappings = 0;
    u32 tileUnmappings = 0;
    /// Frames that ran out of upload space and had to flush and wait.
    u32 uploadFlushes = 0;
};

class GpuTexturePool final : public TextureUploadSink
{
public:
    /// Wraps a texture's real source so the streamer plans in the driver's
    /// tiles. Owned by the pool; lives until `RemoveTexture`.
    class TiledSource final : public TextureMipSource
    {
    public:
        TiledSource(TextureMipSource& inner, const rhi::TextureTiling& tiling) noexcept
            : m_inner(&inner), m_tiling(tiling)
        {
        }
        [[nodiscard]] u32 LevelCount() const noexcept override;
        [[nodiscard]] StreamingLevelInfo Level(u32 level) const noexcept override;
        [[nodiscard]] bool ReadLevel(u32 level, memory::Span<Byte> destination) noexcept override;
        [[nodiscard]] u32 PageCount(u32 firstLevel, u32 levelCount) const noexcept override;
        [[nodiscard]] u32 TailLevel() const noexcept override;

    private:
        TextureMipSource* m_inner = nullptr;
        rhi::TextureTiling m_tiling{};
    };

    GpuTexturePool() noexcept = default;
    ~GpuTexturePool() override;
    GpuTexturePool(const GpuTexturePool&) = delete;
    GpuTexturePool& operator=(const GpuTexturePool&) = delete;

    /// Creates the tile heap, the upload buffers, and a fence. Unsupported
    /// without `Feature::ReservedTextures`.
    [[nodiscard]] core::Status<core::Error> Initialize(rhi::Device& device,
                                                       const GpuTexturePoolDesc& desc,
                                                       memory::Allocator& allocator) noexcept;
    /// Waits for the GPU, then releases everything. Called by the destructor.
    void Shutdown() noexcept;

    /// Creates the reserved texture for `source` (format, extent and mip count
    /// from `desc`; `reserved` and the copy usages are added) and registers it
    /// with `streamer` through a `TiledSource`.
    [[nodiscard]] core::Result<TextureHandle, core::Error> AddTexture(TextureStreamer& streamer,
                                                                      TextureMipSource& source,
                                                                      const rhi::TextureDesc& desc,
                                                                      const char* name) noexcept;
    /// Unregisters from `streamer` (which evicts through this pool), then
    /// destroys the reserved texture once the GPU is done with it.
    void RemoveTexture(TextureStreamer& streamer, TextureHandle texture) noexcept;

    /// Starts a frame: waits until this frame's upload buffer is free and opens
    /// a command list.
    [[nodiscard]] core::Status<core::Error> BeginFrame() noexcept;
    /// Submits the frame's copies on the pool's queue and signals the fence.
    [[nodiscard]] core::Status<core::Error> EndFrame() noexcept;

    /// The reserved texture behind a streamed texture, for views and sampling.
    [[nodiscard]] rhi::TextureHandle Texture(TextureHandle texture) const noexcept;
    [[nodiscard]] rhi::TileHeapHandle Heap() const noexcept;
    [[nodiscard]] const GpuTexturePoolStats& Stats() const noexcept;

    void UploadLevels(TextureHandle texture, u32 firstLevel, u32 levelCount,
                      memory::Span<const Byte> data,
                      memory::Span<const u32> pages) noexcept override;
    void EvictLevel(TextureHandle texture, u32 level,
                    memory::Span<const u32> pages) noexcept override;

private:
    struct Impl;
    Impl* m_impl = nullptr;
};

}  // namespace cyber::renderer
