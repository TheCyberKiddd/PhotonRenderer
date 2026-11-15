#include <cyber/core/log.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/renderer/gpu_virtual_texture_feedback.h>
#include <cyber/rhi/resource_state.h>

#include <algorithm>
#include <new>
#include <utility>

namespace cyber::renderer
{

namespace
{

constexpr u32 kMaxFramesInFlight = 8;

core::Error MakeError(core::ErrorCode code, const char* message) noexcept
{
    return core::Error{code, message};
}

// The page a sample asks for is chosen exactly as `VirtualTexture::RequestSample`
// chooses it; the counter index is the page's position in the same flat order
// the CPU uses (mip 0 first, rows, then columns).
const char* const kHlsl = R"hlsl(
void VtFeedback(VtParams p, float2 uv, float lod)
{
    float level = isfinite(lod) ? max(lod, 0.0f) : 0.0f;
    uint mip = min((uint)level, p.mipCount - 1u);
    float2 f = uv - floor(uv);
    uint index = 0u;
    for (uint m = 0u; m < mip; ++m)
    {
        index += max((p.width >> m) / p.pageSize, 1u) * max((p.height >> m) / p.pageSize, 1u);
    }
    uint2 pages = uint2(max((p.width >> mip) / p.pageSize, 1u),
                        max((p.height >> mip) / p.pageSize, 1u));
    uint2 page = min((uint2)(f * (float2)pages), pages - 1u);
    InterlockedAdd(g_vtFeedback[index + page.y * pages.x + page.x], 1u);
}
)hlsl";

}  // namespace

struct GpuVirtualTextureFeedback::Impl
{
    explicit Impl(memory::Allocator& allocator) noexcept
        : allocator(&allocator), mipOffsets(allocator), counts(allocator)
    {
    }

    struct Pending
    {
        u32 slot = 0;
        rhi::FenceHandle fence{};
        u64 value = 0;
    };

    memory::Allocator* allocator = nullptr;
    rhi::Device* device = nullptr;
    VirtualTexture* texture = nullptr;
    GpuVirtualTextureFeedbackDesc desc{};
    u32 totalPages = 0;
    u64 bytes = 0;
    memory::DynamicArray<u32> mipOffsets;
    rhi::BufferHandle buffer{};
    rhi::BufferHandle zeros{};
    rhi::BufferHandle readback[kMaxFramesInFlight] = {};
    Pending pending[kMaxFramesInFlight] = {};
    u32 head = 0;
    u32 pendingCount = 0;
    u32 nextSlot = 0;
    memory::DynamicArray<u32> counts;
    GpuVirtualTextureFeedbackStats stats{};

    [[nodiscard]] bool MakeBuffer(rhi::BufferUsage usage, u32 stride, const char* label,
                                  rhi::BufferHandle& out) noexcept
    {
        rhi::BufferDesc bufferDesc{};
        bufferDesc.usage = usage;
        bufferDesc.sizeBytes = bytes;
        bufferDesc.elementStride = stride;
        bufferDesc.label = label;
        auto created = device->CreateBuffer(bufferDesc);
        if (!created)
        {
            CG_LOG(Error, Renderer, "virtual texture feedback: buffer '%s' failed: %s", label,
                   created.Error().message);
            return false;
        }
        out = created.Value();
        return true;
    }

    // Reads the oldest in-flight frame (its fence has been checked) and turns its
    // counters into page requests.
    u32 ProcessOldest() noexcept
    {
        const Pending frame = pending[head];
        head = (head + 1u) % desc.framesInFlight;
        --pendingCount;
        if (!device
                 ->ReadBuffer(readback[frame.slot], 0,
                              memory::Span<Byte>(reinterpret_cast<Byte*>(counts.Data()),
                                                 static_cast<usize>(bytes)))
                 .IsOk())
        {
            CG_LOG(Error, Renderer, "virtual texture feedback: readback failed");
            return 0;
        }
        stats.readbackBytes += bytes;
        u32 requested = 0;
        u64 samples = 0;
        const u32 mips = texture->MipCount();
        for (u32 mip = 0; mip < mips; ++mip)
        {
            const u32 pagesX = texture->PagesX(mip);
            const u32 pagesY = texture->PagesY(mip);
            const u32* row = counts.Data() + mipOffsets[mip];
            for (u32 y = 0; y < pagesY; ++y)
            {
                for (u32 x = 0; x < pagesX; ++x)
                {
                    const u32 count = row[y * pagesX + x];
                    if (count != 0)
                    {
                        samples += count;
                        texture->RequestPage(mip, x, y);
                        ++requested;
                    }
                }
            }
        }
        ++stats.resolvedFrames;
        stats.pagesLastFrame = requested;
        stats.samplesLastFrame = samples;
        stats.totalPageRequests += requested;
        return requested;
    }
};

GpuVirtualTextureFeedback::~GpuVirtualTextureFeedback()
{
    Shutdown();
}

const char* GpuVirtualTextureFeedback::ShaderSource() noexcept
{
    return kHlsl;
}

core::Status<core::Error>
GpuVirtualTextureFeedback::Initialize(rhi::Device& device, VirtualTexture& texture,
                                      const GpuVirtualTextureFeedbackDesc& desc,
                                      memory::Allocator& allocator) noexcept
{
    using StatusType = core::Status<core::Error>;
    Shutdown();
    if (desc.framesInFlight == 0 || desc.framesInFlight > kMaxFramesInFlight ||
        texture.MipCount() == 0)
    {
        return StatusType::Err(
            MakeError(core::ErrorCode::InvalidArgument, "invalid feedback description"));
    }
    void* memory = allocator.Allocate(sizeof(Impl), alignof(Impl));
    if (memory == nullptr)
    {
        return StatusType::Err(MakeError(core::ErrorCode::Internal, "out of memory"));
    }
    m_impl = new (memory) Impl(allocator);
    Impl& impl = *m_impl;
    impl.device = &device;
    impl.texture = &texture;
    impl.desc = desc;

    u64 total = 0;
    for (u32 mip = 0; mip < texture.MipCount(); ++mip)
    {
        if (!impl.mipOffsets.TryPushBack(static_cast<u32>(total)))
        {
            Shutdown();
            return StatusType::Err(MakeError(core::ErrorCode::Internal, "out of memory"));
        }
        total += static_cast<u64>(texture.PagesX(mip)) * texture.PagesY(mip);
    }
    // The shader's index arithmetic and the readback assume the mip chain halves.
    for (u32 mip = 0; mip < texture.MipCount(); ++mip)
    {
        if (texture.PagesX(mip) != std::max(texture.PagesX(0) >> mip, 1u) ||
            texture.PagesY(mip) != std::max(texture.PagesY(0) >> mip, 1u))
        {
            Shutdown();
            return StatusType::Err(MakeError(core::ErrorCode::InvalidArgument,
                                             "page mips do not halve; feedback layout differs"));
        }
    }
    if (total == 0 || total > (1u << 26u))
    {
        Shutdown();
        return StatusType::Err(MakeError(core::ErrorCode::OutOfRange, "page count out of range"));
    }
    impl.totalPages = static_cast<u32>(total);
    impl.bytes = total * sizeof(u32);
    if (!impl.counts.TryResize(static_cast<usize>(total)))
    {
        Shutdown();
        return StatusType::Err(MakeError(core::ErrorCode::Internal, "out of memory"));
    }

    bool ok = impl.MakeBuffer(rhi::BufferUsage::Storage | rhi::BufferUsage::CopyDestination |
                                  rhi::BufferUsage::CopySource,
                              4, "vt_feedback_counters", impl.buffer) &&
              impl.MakeBuffer(rhi::BufferUsage::HostVisible | rhi::BufferUsage::CopySource, 0,
                              "vt_feedback_zeros", impl.zeros);
    for (u32 slot = 0; ok && slot < desc.framesInFlight; ++slot)
    {
        ok = impl.MakeBuffer(rhi::BufferUsage::Readback | rhi::BufferUsage::CopyDestination, 0,
                             "vt_feedback_readback", impl.readback[slot]);
    }
    if (ok)
    {
        // The zero source is written once; `counts` is zero-initialised by the
        // dynamic array's resize and doubles as the zero image.
        for (usize i = 0; i < impl.counts.Size(); ++i)
        {
            impl.counts[i] = 0;
        }
        ok = device
                 .WriteBuffer(
                     impl.zeros, 0,
                     memory::Span<const Byte>(reinterpret_cast<const Byte*>(impl.counts.Data()),
                                              static_cast<usize>(impl.bytes)))
                 .IsOk();
    }
    if (!ok)
    {
        Shutdown();
        return StatusType::Err(MakeError(core::ErrorCode::Internal, "feedback buffers failed"));
    }
    return StatusType::Ok();
}

void GpuVirtualTextureFeedback::Shutdown() noexcept
{
    if (m_impl == nullptr)
    {
        return;
    }
    Impl& impl = *m_impl;
    if (impl.device != nullptr)
    {
        // Frames still in flight are being copied into the readback ring.
        (void)impl.device->WaitIdle();
        if (impl.buffer.IsValid())
        {
            (void)impl.device->DestroyBuffer(impl.buffer);
        }
        if (impl.zeros.IsValid())
        {
            (void)impl.device->DestroyBuffer(impl.zeros);
        }
        for (const rhi::BufferHandle slot : impl.readback)
        {
            if (slot.IsValid())
            {
                (void)impl.device->DestroyBuffer(slot);
            }
        }
    }
    memory::Allocator* allocator = impl.allocator;
    impl.~Impl();
    allocator->Deallocate(m_impl);
    m_impl = nullptr;
}

core::Status<core::Error> GpuVirtualTextureFeedback::Begin(rhi::CommandList& list) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (m_impl == nullptr)
    {
        return StatusType::Err(MakeError(core::ErrorCode::InvalidArgument, "not initialized"));
    }
    Impl& impl = *m_impl;
    rhi::BufferBarrier toClear[2];
    u32 barrierCount = 0;
    const rhi::ResourceState counterState = impl.device->BufferStateOf(impl.buffer);
    if (counterState != rhi::ResourceState::CopyDestination)
    {
        toClear[barrierCount++] = {impl.buffer, counterState, rhi::ResourceState::CopyDestination};
    }
    const rhi::ResourceState zeroState = impl.device->BufferStateOf(impl.zeros);
    if (zeroState != rhi::ResourceState::CopySource)
    {
        toClear[barrierCount++] = {impl.zeros, zeroState, rhi::ResourceState::CopySource};
    }
    auto status = list.Barrier(memory::Span<const rhi::BufferBarrier>(toClear, barrierCount));
    if (!status)
    {
        return status;
    }
    rhi::BufferCopy copy{};
    copy.source = impl.zeros;
    copy.destination = impl.buffer;
    copy.sizeBytes = impl.bytes;
    status = list.CopyBuffer(copy);
    if (!status)
    {
        return status;
    }
    const rhi::BufferBarrier toShader{impl.buffer, rhi::ResourceState::CopyDestination,
                                      rhi::ResourceState::ShaderWrite};
    return list.Barrier(memory::Span<const rhi::BufferBarrier>(&toShader, 1));
}

core::Status<core::Error> GpuVirtualTextureFeedback::End(rhi::CommandList& list,
                                                         rhi::FenceHandle fence,
                                                         u64 fenceValue) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (m_impl == nullptr)
    {
        return StatusType::Err(MakeError(core::ErrorCode::InvalidArgument, "not initialized"));
    }
    Impl& impl = *m_impl;
    if (impl.pendingCount == impl.desc.framesInFlight)
    {
        // Every slot is in flight: the oldest has to finish before it is reused.
        ++impl.stats.stalls;
        const Impl::Pending& oldest = impl.pending[impl.head];
        const auto waited = impl.device->WaitFence(oldest.fence, oldest.value);
        if (!waited)
        {
            return waited;
        }
        (void)impl.ProcessOldest();
    }
    const u32 slot = impl.nextSlot;
    impl.nextSlot = (impl.nextSlot + 1u) % impl.desc.framesInFlight;

    rhi::BufferBarrier toCopy[2];
    u32 barrierCount = 0;
    toCopy[barrierCount++] = {impl.buffer, rhi::ResourceState::ShaderWrite,
                              rhi::ResourceState::CopySource};
    const rhi::ResourceState slotState = impl.device->BufferStateOf(impl.readback[slot]);
    if (slotState != rhi::ResourceState::CopyDestination)
    {
        toCopy[barrierCount++] = {impl.readback[slot], slotState,
                                  rhi::ResourceState::CopyDestination};
    }
    auto status = list.Barrier(memory::Span<const rhi::BufferBarrier>(toCopy, barrierCount));
    if (!status)
    {
        return status;
    }
    rhi::BufferCopy copy{};
    copy.source = impl.buffer;
    copy.destination = impl.readback[slot];
    copy.sizeBytes = impl.bytes;
    status = list.CopyBuffer(copy);
    if (!status)
    {
        return status;
    }
    const u32 tail = (impl.head + impl.pendingCount) % impl.desc.framesInFlight;
    impl.pending[tail] = Impl::Pending{slot, fence, fenceValue};
    ++impl.pendingCount;
    return StatusType::Ok();
}

u32 GpuVirtualTextureFeedback::Resolve() noexcept
{
    if (m_impl == nullptr)
    {
        return 0;
    }
    Impl& impl = *m_impl;
    u32 requested = 0;
    while (impl.pendingCount > 0)
    {
        const Impl::Pending& oldest = impl.pending[impl.head];
        // A poll: not finished yet (or an unusable fence) leaves it for later.
        if (!impl.device->WaitFence(oldest.fence, oldest.value, 0).IsOk())
        {
            break;
        }
        requested += impl.ProcessOldest();
    }
    return requested;
}

rhi::BufferHandle GpuVirtualTextureFeedback::Buffer() const noexcept
{
    return m_impl != nullptr ? m_impl->buffer : rhi::BufferHandle{};
}

u32 GpuVirtualTextureFeedback::TotalPages() const noexcept
{
    return m_impl != nullptr ? m_impl->totalPages : 0;
}

memory::Span<const u32> GpuVirtualTextureFeedback::LastCounts() const noexcept
{
    return m_impl != nullptr ? memory::Span<const u32>(m_impl->counts.Data(), m_impl->counts.Size())
                             : memory::Span<const u32>{};
}

const GpuVirtualTextureFeedbackStats& GpuVirtualTextureFeedback::Stats() const noexcept
{
    static const GpuVirtualTextureFeedbackStats kEmpty{};
    return m_impl != nullptr ? m_impl->stats : kEmpty;
}

}  // namespace cyber::renderer
