#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/span.h>
#include <cyber/renderer/virtual_texture.h>
#include <cyber/rhi/command_list.h>
#include <cyber/rhi/device.h>

namespace cyber::renderer
{

/// GPU-driven page feedback for virtual texturing (Phase 33).
///
/// A virtual texture streams what the view samples, so the renderer has to say
/// which pages that is. On the CPU a sampler calls `VirtualTexture::RequestSample`;
/// a GPU renderer cannot, so the shader itself records the request. This is the
/// scheme production virtual-texturing systems use on every GPU: the shader that
/// samples the texture also bumps a counter for the page it asked for, in a
/// small buffer, and the CPU reads that buffer back a few frames later and
/// turns every non-zero entry into `RequestPage`.
///
/// It does not need hardware sampler feedback (`D3D12_FEATURE_DATA_D3D12_OPTIONS7`
/// `SamplerFeedbackTier`, which neither the Intel HD Graphics 530 nor WARP
/// report), and it works on every backend that has storage buffers and atomics.
/// Where hardware sampler feedback exists it would produce the same request set
/// through the same `RequestPage` calls; the readback and the streaming above
/// this class would not change.
///
/// Per frame:
///   1. `Begin(list)`  -- records the clear of the counter buffer.
///   2. The frame's shaders call `VtFeedback(p, uv, lod)` (see `ShaderSource`)
///      next to every `VtSample`, with `Buffer()` bound as a storage buffer.
///   3. `End(list, fence, value)` -- records the copy of the counters into the
///      next slot of a small readback ring. The caller then submits the list and
///      signals `fence` with `value`.
///   4. `Resolve()` -- any time later, polls the ring without blocking; each
///      frame the GPU has finished is read back and its pages requested on the
///      texture. The latency is the frames in flight, and the requests of an
///      older frame are as good as a new one: pages stay requested until the
///      streamer has them.
struct GpuVirtualTextureFeedbackDesc
{
    /// Readback slots in rotation: how many frames of feedback may be in flight.
    u32 framesInFlight = 3;
    const char* label = "virtual_texture_feedback";
};

struct GpuVirtualTextureFeedbackStats
{
    /// Frames read back and turned into requests.
    u32 resolvedFrames = 0;
    /// Frames `End` had to wait for because the ring was full.
    u32 stalls = 0;
    /// Distinct pages requested by the most recent resolved frame.
    u32 pagesLastFrame = 0;
    /// Samples (counter total) the most recent resolved frame recorded.
    u64 samplesLastFrame = 0;
    u64 totalPageRequests = 0;
    u64 readbackBytes = 0;
};

class GpuVirtualTextureFeedback
{
public:
    GpuVirtualTextureFeedback() noexcept = default;
    ~GpuVirtualTextureFeedback();
    GpuVirtualTextureFeedback(const GpuVirtualTextureFeedback&) = delete;
    GpuVirtualTextureFeedback& operator=(const GpuVirtualTextureFeedback&) = delete;

    /// Creates the counter buffer (one 32-bit counter per page of every mip of
    /// `texture`), a zero source for clearing it and the readback ring.
    /// `texture` receives the requests and must outlive this object.
    [[nodiscard]] core::Status<core::Error> Initialize(rhi::Device& device, VirtualTexture& texture,
                                                       const GpuVirtualTextureFeedbackDesc& desc,
                                                       memory::Allocator& allocator) noexcept;
    /// Waits for the GPU and releases everything. Called by the destructor.
    void Shutdown() noexcept;

    /// Records the clear of the counters and leaves the buffer writable by
    /// shaders (`ResourceState::ShaderWrite`).
    [[nodiscard]] core::Status<core::Error> Begin(rhi::CommandList& list) noexcept;
    /// Records the copy of the counters into the next readback slot. The caller
    /// submits `list` and signals `fence` with `fenceValue` when it completes;
    /// `Resolve` reads the slot once the fence has reached that value. When every
    /// slot is still in flight, waits for the oldest one first.
    [[nodiscard]] core::Status<core::Error> End(rhi::CommandList& list, rhi::FenceHandle fence,
                                                u64 fenceValue) noexcept;
    /// Reads back every finished frame, oldest first, and requests the pages
    /// they counted. Never blocks. Returns the pages requested.
    u32 Resolve() noexcept;

    /// The counter buffer: `TotalPages()` 32-bit values, bound by shaders as a
    /// read-write structured buffer of `uint`.
    [[nodiscard]] rhi::BufferHandle Buffer() const noexcept;
    [[nodiscard]] u32 TotalPages() const noexcept;
    /// The counters of the most recently resolved frame, in page order (mip 0
    /// first, rows then columns within a mip).
    [[nodiscard]] memory::Span<const u32> LastCounts() const noexcept;
    [[nodiscard]] const GpuVirtualTextureFeedbackStats& Stats() const noexcept;

    /// HLSL defining `void VtFeedback(VtParams p, float2 uv, float lod)`, which
    /// adds one to the counter of the page `VirtualTexture::RequestSample(uv,
    /// lod)` would request. It reads the global `RWStructuredBuffer<uint>
    /// g_vtFeedback` that the including shader declares and binds, and uses
    /// `VtParams`, so it goes after `GpuVirtualTexture::ShaderSource()`. Static
    /// lifetime.
    [[nodiscard]] static const char* ShaderSource() noexcept;

private:
    struct Impl;
    Impl* m_impl = nullptr;
};

}  // namespace cyber::renderer
