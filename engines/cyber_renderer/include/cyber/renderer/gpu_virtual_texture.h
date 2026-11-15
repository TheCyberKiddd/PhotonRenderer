#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/memory/allocator.h>
#include <cyber/renderer/virtual_texture.h>
#include <cyber/rhi/command_list.h>
#include <cyber/rhi/device.h>
#include <cyber/rhi/texture.h>

namespace cyber::renderer
{

/// The GPU side of virtual texturing (Phase 33): mirrors a `VirtualTexture`'s
/// page table and cache atlas into RHI textures and ships the HLSL that
/// samples them, so a shader resolves a virtual (u, v, lod) exactly as
/// `VirtualTexture::Sample` does on the CPU.
///
/// - The page table is one R32_UINT texture with a full mip chain: mip m holds
///   the page-table entries of virtual mip m in `EncodePageTableEntry` form.
/// - The atlas is one RGBA8 texture, `AtlasWidth() x AtlasHeight()`.
/// - `Sync` runs after `VirtualTexture::Update` and uploads only what changed:
///   dirty page-table mips and the cache pages the update wrote. The first
///   `Sync` uploads every page-table mip and every cache page a valid entry
///   points at (the pinned top level).
/// - Sampling clamps to the finest resident ancestor, so a shader never reads
///   an unloaded page.
struct GpuVirtualTextureDesc
{
    /// Upload buffers in rotation; a buffer is reused once the sync that used
    /// it has finished on the GPU.
    u32 framesInFlight = 3;
    rhi::QueueKind queue = rhi::QueueKind::Graphics;
    const char* label = "virtual_texture";
};

struct GpuVirtualTextureStats
{
    u64 uploadedBytes = 0;
    u32 syncs = 0;
    u32 pageTableUploads = 0;  // mips uploaded
    u32 pageUploads = 0;       // cache pages uploaded
    /// Syncs that ran out of upload space and had to flush and wait.
    u32 uploadFlushes = 0;
};

/// Constants a shader needs, laid out as the HLSL `VtParams` struct (six
/// 32-bit words, padded to 8).
struct GpuVirtualTextureParams
{
    u32 width = 0;
    u32 height = 0;
    u32 pageSize = 0;
    u32 border = 0;
    u32 stride = 0;
    u32 mipCount = 0;
    u32 reserved0 = 0;
    u32 reserved1 = 0;
};

class GpuVirtualTexture
{
public:
    GpuVirtualTexture() noexcept = default;
    ~GpuVirtualTexture();
    GpuVirtualTexture(const GpuVirtualTexture&) = delete;
    GpuVirtualTexture& operator=(const GpuVirtualTexture&) = delete;

    /// Creates the page-table and atlas textures, their views, the upload
    /// buffers and a fence, sized from `source`. `source` is only read here
    /// and in `Sync`.
    [[nodiscard]] core::Status<core::Error> Initialize(rhi::Device& device,
                                                       const VirtualTexture& source,
                                                       const GpuVirtualTextureDesc& desc,
                                                       memory::Allocator& allocator) noexcept;
    /// Waits for the GPU, then releases everything. Called by the destructor.
    void Shutdown() noexcept;

    /// Uploads what the last `VirtualTexture::Update` changed (everything
    /// resident on the first call) and submits it on the queue. Call once per
    /// frame, after `Update` and before the frame's draws. Pass the same
    /// texture given to `Initialize`.
    [[nodiscard]] core::Status<core::Error> Sync(const VirtualTexture& source) noexcept;

    [[nodiscard]] rhi::TextureHandle PageTable() const noexcept;
    [[nodiscard]] rhi::TextureHandle Atlas() const noexcept;
    [[nodiscard]] rhi::TextureViewHandle PageTableView() const noexcept;
    [[nodiscard]] rhi::TextureViewHandle AtlasView() const noexcept;
    [[nodiscard]] GpuVirtualTextureParams Params() const noexcept;
    [[nodiscard]] const GpuVirtualTextureStats& Stats() const noexcept;

    /// HLSL source defining `struct VtParams` and
    /// `float4 VtSample(VtParams p, float2 uv, float lod)`, which reads the globals
    /// `Texture2D<uint> g_vtPageTable` and `Texture2D<float4> g_vtAtlas` that the
    /// including shader declares and binds *before* this source; `p` comes from
    /// `Params()`. The
    /// lookup and bilinear filter mirror `VirtualTexture::Sample`. Static
    /// lifetime.
    [[nodiscard]] static const char* ShaderSource() noexcept;

private:
    struct Impl;
    Impl* m_impl = nullptr;
};

}  // namespace cyber::renderer
