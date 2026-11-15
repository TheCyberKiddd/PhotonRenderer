#include <cyber/core/log.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/renderer/gpu_virtual_texture.h>
#include <cyber/rhi/resource_state.h>
#include <cyber/rhi/submission.h>

#include <algorithm>
#include <cstring>
#include <new>
#include <utility>

namespace cyber::renderer
{

namespace
{

// The strictest buffer-texture copy placement any backend imposes (D3D12's).
constexpr u64 kUploadOffsetAlignment = 512;
constexpr u64 kUploadRowAlignment = 256;
constexpr u32 kMaxFramesInFlight = 4;
// Pages the upload buffer holds beyond the page tables; a sync that needs more
// flushes and continues.
constexpr u32 kUploadPageSlack = 32;

u64 AlignUp(u64 value, u64 alignment) noexcept
{
    return (value + alignment - 1) / alignment * alignment;
}

core::Error MakeError(core::ErrorCode code, const char* message) noexcept
{
    return core::Error{code, message};
}

const char* const kHlsl = R"hlsl(
struct VtParams
{
    uint width;
    uint height;
    uint pageSize;
    uint border;
    uint stride;
    uint mipCount;
    uint reserved0;
    uint reserved1;
};

float4 VtSample(VtParams p, float2 uv, float lod)
{
    float level = isfinite(lod) ? max(lod, 0.0f) : 0.0f;
    uint mip = min((uint)level, p.mipCount - 1u);
    float2 f = uv - floor(uv);
    uint2 pages = uint2(max((p.width >> mip) / p.pageSize, 1u),
                        max((p.height >> mip) / p.pageSize, 1u));
    uint2 page = min((uint2)(f * (float2)pages), pages - 1u);
    uint entry = g_vtPageTable.Load(int3((int2)page, (int)mip));
    uint2 cache = uint2(entry & 0x3FFu, (entry >> 10u) & 0x3FFu);
    uint residentMip = (entry >> 20u) & 0x1Fu;
    float2 texel = f * float2((float)(p.width >> residentMip), (float)(p.height >> residentMip));
    float pageSize = (float)p.pageSize;
    float2 inPage = texel - floor(texel / pageSize) * pageSize;
    float2 a = (float2)(cache * p.stride) + (float)p.border + inPage - 0.5f;
    float2 base = floor(a);
    float2 w = a - base;
    int2 x0 = (int2)base;
    float4 t00 = g_vtAtlas.Load(int3(x0, 0));
    float4 t10 = g_vtAtlas.Load(int3(x0 + int2(1, 0), 0));
    float4 t01 = g_vtAtlas.Load(int3(x0 + int2(0, 1), 0));
    float4 t11 = g_vtAtlas.Load(int3(x0 + int2(1, 1), 0));
    return lerp(lerp(t00, t10, w.x), lerp(t01, t11, w.x), w.y);
}
)hlsl";

}  // namespace

struct GpuVirtualTexture::Impl
{
    explicit Impl(memory::Allocator& allocator) noexcept
        : allocator(&allocator), staging(allocator), pageSeen(allocator)
    {
    }

    memory::Allocator* allocator = nullptr;
    rhi::Device* device = nullptr;
    GpuVirtualTextureDesc desc{};
    GpuVirtualTextureParams params{};
    u32 cachePagesX = 0;
    u32 cachePagesY = 0;
    u64 uploadBytes = 0;
    rhi::TextureHandle pageTable{};
    rhi::TextureHandle atlas{};
    rhi::TextureViewHandle pageTableView{};
    rhi::TextureViewHandle atlasView{};
    rhi::BufferHandle uploads[kMaxFramesInFlight] = {};
    u64 uploadFenceValues[kMaxFramesInFlight] = {};
    rhi::FenceHandle fence{};
    u64 fenceValue = 0;
    u32 frame = 0;
    u64 uploadOffset = 0;
    rhi::CommandList list;
    bool listOpen = false;
    bool synced = false;
    // Submitted lists stay alive until their fence passes.
    rhi::CommandList submitted[kMaxFramesInFlight];
    memory::DynamicArray<Byte> staging;
    memory::DynamicArray<Byte> pageSeen;
    GpuVirtualTextureStats stats{};

    [[nodiscard]] bool OpenList() noexcept
    {
        rhi::CommandListDesc listDesc{};
        listDesc.queue = desc.queue;
        auto created = device->CreateCommandList(listDesc);
        if (!created || !created.Value().Begin().IsOk())
        {
            return false;
        }
        list = std::move(created.Value());
        listOpen = true;
        return true;
    }

    [[nodiscard]] bool Submit() noexcept
    {
        if (!listOpen)
        {
            return true;
        }
        listOpen = false;
        if (!list.End().IsOk())
        {
            return false;
        }
        const rhi::CommandListHandle handle = list.Handle();
        const rhi::FenceSignal signal{fence, ++fenceValue};
        rhi::SubmitDesc submit{};
        submit.queue = desc.queue;
        submit.commandLists = memory::Span<const rhi::CommandListHandle>(&handle, 1);
        submit.signals = memory::Span<const rhi::FenceSignal>(&signal, 1);
        const bool ok = device->Submit(submit).IsOk();
        if (uploadFenceValues[frame] != 0)
        {
            (void)device->WaitFence(fence, uploadFenceValues[frame]);
        }
        submitted[frame] = std::move(list);
        uploadFenceValues[frame] = fenceValue;
        list = rhi::CommandList{};
        return ok;
    }

    [[nodiscard]] bool Flush() noexcept
    {
        ++stats.uploadFlushes;
        if (!Submit() || !device->WaitFence(fence, fenceValue).IsOk())
        {
            return false;
        }
        uploadOffset = 0;
        return OpenList();
    }

    // Copies a `width x height` image of `rowBytes`-byte rows, row r read from
    // `source + r * sourceStride`, into (x, y) of `texture` mip `mip`.
    [[nodiscard]] bool CopyImage(rhi::TextureHandle texture, u32 mip, u32 x, u32 y, u32 width,
                                 u32 height, u64 rowBytes, const Byte* source,
                                 u64 sourceStride) noexcept
    {
        const u64 pitch = AlignUp(rowBytes, kUploadRowAlignment);
        const u64 imageBytes = pitch * height;
        u64 offset = AlignUp(uploadOffset, kUploadOffsetAlignment);
        if (offset + imageBytes > uploadBytes)
        {
            if (!Flush())
            {
                return false;
            }
            offset = 0;
            if (imageBytes > uploadBytes)
            {
                return false;
            }
        }
        if (!staging.TryResize(static_cast<usize>(imageBytes)))
        {
            return false;
        }
        for (u32 row = 0; row < height; ++row)
        {
            std::memcpy(staging.Data() + row * pitch, source + row * sourceStride,
                        static_cast<usize>(rowBytes));
        }
        const rhi::BufferHandle upload = uploads[frame];
        const auto written = device->WriteBuffer(
            upload, offset, memory::Span<const Byte>(staging.Data(), staging.Size()));
        if (!written)
        {
            CG_LOG(Error, Renderer, "virtual texture: upload write failed: %s",
                   written.Error().message);
            return false;
        }
        uploadOffset = offset + imageBytes;

        const rhi::ResourceState before = device->TextureStateOf(texture, mip, 0);
        rhi::SubresourceRange range{};
        range.baseMipLevel = mip;
        range.mipLevelCount = 1;
        range.layerCount = 1;
        rhi::BufferBarrier bufferBarrier{upload, device->BufferStateOf(upload),
                                         rhi::ResourceState::CopySource};
        const bool bufferReady = bufferBarrier.from == rhi::ResourceState::CopySource;
        const rhi::TextureBarrier toCopy{texture, range, before,
                                         rhi::ResourceState::CopyDestination};
        const auto toCopyStatus = list.Barrier(
            memory::Span<const rhi::BufferBarrier>(&bufferBarrier, bufferReady ? 0u : 1u),
            memory::Span<const rhi::TextureBarrier>(&toCopy, 1));
        if (!toCopyStatus)
        {
            CG_LOG(Error, Renderer, "virtual texture: copy barrier failed: %s",
                   toCopyStatus.Error().message);
            return false;
        }
        rhi::BufferTextureCopy copy{};
        copy.buffer = upload;
        copy.bufferOffset = offset;
        copy.bufferRowPitch = static_cast<u32>(pitch);
        copy.texture = texture;
        copy.mipLevel = mip;
        copy.x = x;
        copy.y = y;
        copy.width = width;
        copy.height = height;
        copy.depth = 1;
        const auto copied = list.CopyBufferToTexture(copy);
        if (!copied)
        {
            CG_LOG(Error, Renderer, "virtual texture: copy to mip %u failed: %s", mip,
                   copied.Error().message);
            return false;
        }
        const rhi::TextureBarrier toSample{texture, range, rhi::ResourceState::CopyDestination,
                                           rhi::ResourceState::ShaderRead};
        const auto toSampleStatus =
            list.Barrier(memory::Span<const rhi::BufferBarrier>{},
                         memory::Span<const rhi::TextureBarrier>(&toSample, 1));
        if (!toSampleStatus)
        {
            CG_LOG(Error, Renderer, "virtual texture: sample barrier failed: %s",
                   toSampleStatus.Error().message);
            return false;
        }
        return true;
    }

    [[nodiscard]] bool UploadPageTable(const VirtualTexture& source, u32 mip) noexcept
    {
        const memory::Span<const u32> table = source.PageTable(mip);
        const u32 pagesX = source.PagesX(mip);
        const u32 pagesY = source.PagesY(mip);
        if (table.Size() != static_cast<usize>(pagesX) * pagesY)
        {
            return false;
        }
        if (!CopyImage(pageTable, mip, 0, 0, pagesX, pagesY, static_cast<u64>(pagesX) * 4u,
                       reinterpret_cast<const Byte*>(table.Data()), static_cast<u64>(pagesX) * 4u))
        {
            return false;
        }
        ++stats.pageTableUploads;
        stats.uploadedBytes += static_cast<u64>(pagesX) * pagesY * 4u;
        return true;
    }

    [[nodiscard]] bool UploadCachePage(const VirtualTexture& source, u32 cacheX,
                                       u32 cacheY) noexcept
    {
        const u32 stride = params.stride;
        const u64 atlasRow = static_cast<u64>(source.AtlasWidth()) * 4u;
        const Byte* atlasData = source.Atlas().Data();
        const Byte* first = atlasData + static_cast<u64>(cacheY) * stride * atlasRow +
                            static_cast<u64>(cacheX) * stride * 4u;
        if (!CopyImage(atlas, 0, cacheX * stride, cacheY * stride, stride, stride,
                       static_cast<u64>(stride) * 4u, first, atlasRow))
        {
            return false;
        }
        ++stats.pageUploads;
        stats.uploadedBytes += static_cast<u64>(stride) * stride * 4u;
        return true;
    }
};

GpuVirtualTexture::~GpuVirtualTexture()
{
    Shutdown();
}

const char* GpuVirtualTexture::ShaderSource() noexcept
{
    return kHlsl;
}

core::Status<core::Error> GpuVirtualTexture::Initialize(rhi::Device& device,
                                                        const VirtualTexture& source,
                                                        const GpuVirtualTextureDesc& desc,
                                                        memory::Allocator& allocator) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (m_impl != nullptr)
    {
        return StatusType::Err(
            MakeError(core::ErrorCode::AlreadyExists, "gpu virtual texture is initialized"));
    }
    if (source.MipCount() == 0 || desc.framesInFlight == 0 ||
        desc.framesInFlight > kMaxFramesInFlight)
    {
        return StatusType::Err(
            MakeError(core::ErrorCode::InvalidArgument, "invalid gpu virtual texture description"));
    }
    // The shader derives per-mip page counts as max(pagesX(0) >> mip, 1), the
    // same extents the page-table texture's mip chain has.
    for (u32 mip = 0; mip < source.MipCount(); ++mip)
    {
        if (source.PagesX(mip) != std::max(source.PagesX(0) >> mip, 1u) ||
            source.PagesY(mip) != std::max(source.PagesY(0) >> mip, 1u))
        {
            return StatusType::Err(MakeError(core::ErrorCode::Unsupported,
                                             "page table does not form a halving mip chain"));
        }
    }
    void* memory = allocator.Allocate(sizeof(Impl), alignof(Impl));
    if (memory == nullptr)
    {
        return StatusType::Err(MakeError(core::ErrorCode::OutOfRange, "out of memory"));
    }
    m_impl = new (memory) Impl(allocator);
    Impl& impl = *m_impl;
    impl.device = &device;
    impl.desc = desc;

    const VirtualTextureDesc& vt = source.Desc();
    impl.params.width = vt.width;
    impl.params.height = vt.height;
    impl.params.pageSize = vt.pageSize;
    impl.params.border = vt.border;
    impl.params.stride = source.PageStride();
    impl.params.mipCount = source.MipCount();
    impl.cachePagesX = vt.cachePagesX;
    impl.cachePagesY = vt.cachePagesY;

    const u64 pageBytes =
        AlignUp(AlignUp(static_cast<u64>(impl.params.stride) * 4u, kUploadRowAlignment) *
                    impl.params.stride,
                kUploadOffsetAlignment);
    u64 tableBytes = 0;
    for (u32 mip = 0; mip < source.MipCount(); ++mip)
    {
        tableBytes +=
            AlignUp(AlignUp(static_cast<u64>(source.PagesX(mip)) * 4u, kUploadRowAlignment) *
                        source.PagesY(mip),
                    kUploadOffsetAlignment);
    }
    impl.uploadBytes = tableBytes + pageBytes * kUploadPageSlack;

    const auto fail = [this](const core::Error& error) noexcept
    {
        Shutdown();
        return StatusType::Err(error);
    };

    rhi::TextureDesc tableDesc{};
    tableDesc.format = rhi::Format::R32Uint;
    tableDesc.usage = rhi::TextureUsage::Sampled | rhi::TextureUsage::CopyDestination;
    tableDesc.width = source.PagesX(0);
    tableDesc.height = source.PagesY(0);
    tableDesc.mipLevels = source.MipCount();
    tableDesc.label = "virtual_texture_page_table";
    auto table = device.CreateTexture(tableDesc);
    if (!table)
    {
        return fail(table.Error());
    }
    impl.pageTable = table.Value();

    rhi::TextureDesc atlasDesc{};
    atlasDesc.format = rhi::Format::RGBA8Unorm;
    atlasDesc.usage = rhi::TextureUsage::Sampled | rhi::TextureUsage::CopyDestination;
    atlasDesc.width = source.AtlasWidth();
    atlasDesc.height = source.AtlasHeight();
    atlasDesc.label = "virtual_texture_atlas";
    auto atlasTexture = device.CreateTexture(atlasDesc);
    if (!atlasTexture)
    {
        return fail(atlasTexture.Error());
    }
    impl.atlas = atlasTexture.Value();

    rhi::TextureViewDesc viewDesc{};
    viewDesc.texture = impl.pageTable;
    viewDesc.format = rhi::Format::R32Uint;
    viewDesc.label = "virtual_texture_page_table_view";
    auto tableView = device.CreateTextureView(viewDesc);
    if (!tableView)
    {
        return fail(tableView.Error());
    }
    impl.pageTableView = tableView.Value();
    viewDesc.texture = impl.atlas;
    viewDesc.format = rhi::Format::RGBA8Unorm;
    viewDesc.label = "virtual_texture_atlas_view";
    auto atlasView = device.CreateTextureView(viewDesc);
    if (!atlasView)
    {
        return fail(atlasView.Error());
    }
    impl.atlasView = atlasView.Value();

    for (u32 index = 0; index < desc.framesInFlight; ++index)
    {
        rhi::BufferDesc bufferDesc{};
        bufferDesc.usage = rhi::BufferUsage::HostVisible | rhi::BufferUsage::CopySource;
        bufferDesc.sizeBytes = impl.uploadBytes;
        bufferDesc.label = "virtual_texture_upload";
        auto buffer = device.CreateBuffer(bufferDesc);
        if (!buffer)
        {
            return fail(buffer.Error());
        }
        impl.uploads[index] = buffer.Value();
    }
    auto fence = device.CreateFence(rhi::FenceDesc{});
    if (!fence)
    {
        return fail(fence.Error());
    }
    impl.fence = fence.Value();
    return StatusType::Ok();
}

void GpuVirtualTexture::Shutdown() noexcept
{
    if (m_impl == nullptr)
    {
        return;
    }
    Impl& impl = *m_impl;
    if (impl.listOpen)
    {
        (void)impl.Submit();
    }
    (void)impl.device->WaitIdle();
    for (const rhi::BufferHandle buffer : impl.uploads)
    {
        if (buffer.IsValid())
        {
            (void)impl.device->DestroyBuffer(buffer);
        }
    }
    if (impl.pageTableView.IsValid())
    {
        (void)impl.device->DestroyTextureView(impl.pageTableView);
    }
    if (impl.atlasView.IsValid())
    {
        (void)impl.device->DestroyTextureView(impl.atlasView);
    }
    if (impl.pageTable.IsValid())
    {
        (void)impl.device->DestroyTexture(impl.pageTable);
    }
    if (impl.atlas.IsValid())
    {
        (void)impl.device->DestroyTexture(impl.atlas);
    }
    if (impl.fence.IsValid())
    {
        (void)impl.device->DestroyFence(impl.fence);
    }
    memory::Allocator* allocator = impl.allocator;
    impl.~Impl();
    allocator->Deallocate(m_impl);
    m_impl = nullptr;
}

core::Status<core::Error> GpuVirtualTexture::Sync(const VirtualTexture& source) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (m_impl == nullptr)
    {
        return StatusType::Err(
            MakeError(core::ErrorCode::InvalidArgument, "gpu virtual texture is not initialized"));
    }
    Impl& impl = *m_impl;
    impl.frame = (impl.frame + 1) % impl.desc.framesInFlight;
    if (impl.uploadFenceValues[impl.frame] != 0 &&
        !impl.device->WaitFence(impl.fence, impl.uploadFenceValues[impl.frame]).IsOk())
    {
        return StatusType::Err(MakeError(core::ErrorCode::Internal, "upload fence wait failed"));
    }
    impl.uploadOffset = 0;
    if (!impl.OpenList())
    {
        return StatusType::Err(
            MakeError(core::ErrorCode::Internal, "command list creation failed"));
    }
    const auto failed = [&impl]() noexcept
    {
        (void)impl.Submit();
        return StatusType::Err(
            MakeError(core::ErrorCode::Internal, "virtual texture upload failed"));
    };

    if (!impl.synced)
    {
        // First sync: every page-table mip, and each cache page any valid entry
        // points at (at this point, the pinned top level).
        const usize cachePages = static_cast<usize>(impl.cachePagesX) * impl.cachePagesY;
        if (!impl.pageSeen.TryResize(cachePages))
        {
            return failed();
        }
        std::memset(impl.pageSeen.Data(), 0, cachePages);
        for (u32 mip = 0; mip < source.MipCount(); ++mip)
        {
            if (!impl.UploadPageTable(source, mip))
            {
                return failed();
            }
            const memory::Span<const u32> table = source.PageTable(mip);
            for (usize entryIndex = 0; entryIndex < table.Size(); ++entryIndex)
            {
                const u32 value = table[entryIndex];
                if ((value & 0x80000000u) != 0)
                {
                    const VirtualPageEntry entry = DecodePageTableEntry(value);
                    const usize slot =
                        static_cast<usize>(entry.cacheY) * impl.cachePagesX + entry.cacheX;
                    if (slot < cachePages)
                    {
                        impl.pageSeen[slot] = Byte{1};
                    }
                }
            }
        }
        for (u32 y = 0; y < impl.cachePagesY; ++y)
        {
            for (u32 x = 0; x < impl.cachePagesX; ++x)
            {
                if (static_cast<u32>(impl.pageSeen[static_cast<usize>(y) * impl.cachePagesX + x]) !=
                        0u &&
                    !impl.UploadCachePage(source, x, y))
                {
                    return failed();
                }
            }
        }
        impl.synced = true;
    }
    else
    {
        const memory::Span<const u32> dirty = source.DirtyCachePages();
        for (usize dirtyIndex = 0; dirtyIndex < dirty.Size(); ++dirtyIndex)
        {
            const u32 slot = dirty[dirtyIndex];
            if (!impl.UploadCachePage(source, slot % impl.cachePagesX, slot / impl.cachePagesX))
            {
                return failed();
            }
        }
        for (u32 mip = 0; mip < source.MipCount(); ++mip)
        {
            if (source.PageTableDirty(mip) && !impl.UploadPageTable(source, mip))
            {
                return failed();
            }
        }
    }
    ++impl.stats.syncs;
    return impl.Submit() ? StatusType::Ok()
                         : StatusType::Err(MakeError(core::ErrorCode::Internal, "submit failed"));
}

rhi::TextureHandle GpuVirtualTexture::PageTable() const noexcept
{
    return m_impl != nullptr ? m_impl->pageTable : rhi::TextureHandle{};
}

rhi::TextureHandle GpuVirtualTexture::Atlas() const noexcept
{
    return m_impl != nullptr ? m_impl->atlas : rhi::TextureHandle{};
}

rhi::TextureViewHandle GpuVirtualTexture::PageTableView() const noexcept
{
    return m_impl != nullptr ? m_impl->pageTableView : rhi::TextureViewHandle{};
}

rhi::TextureViewHandle GpuVirtualTexture::AtlasView() const noexcept
{
    return m_impl != nullptr ? m_impl->atlasView : rhi::TextureViewHandle{};
}

GpuVirtualTextureParams GpuVirtualTexture::Params() const noexcept
{
    return m_impl != nullptr ? m_impl->params : GpuVirtualTextureParams{};
}

const GpuVirtualTextureStats& GpuVirtualTexture::Stats() const noexcept
{
    static const GpuVirtualTextureStats empty{};
    return m_impl != nullptr ? m_impl->stats : empty;
}

}  // namespace cyber::renderer
