#include <cyber/core/log.h>
#include <cyber/renderer/gpu_texture_pool.h>
#include <cyber/rhi/format.h>
#include <cyber/rhi/resource_state.h>
#include <cyber/rhi/submission.h>

#include <cstring>
#include <new>
#include <utility>

namespace cyber::renderer
{

namespace
{

// The strictest placement rules any backend imposes on buffer-texture copies
// (D3D12's): each image starts at a 512-byte offset and rows are 256-byte
// aligned. Laying uploads out this way is valid everywhere.
constexpr u64 kUploadOffsetAlignment = 512;
constexpr u64 kUploadRowAlignment = 256;
constexpr u32 kMaxFramesInFlight = 4;

u64 AlignUp(u64 value, u64 alignment) noexcept
{
    return (value + alignment - 1) / alignment * alignment;
}

core::Error MakeError(core::ErrorCode code, const char* message) noexcept
{
    return core::Error{code, message};
}

}  // namespace

// ---------------------------------------------------------------------------
// TiledSource

u32 GpuTexturePool::TiledSource::LevelCount() const noexcept
{
    return m_inner->LevelCount();
}

StreamingLevelInfo GpuTexturePool::TiledSource::Level(u32 level) const noexcept
{
    return m_inner->Level(level);
}

bool GpuTexturePool::TiledSource::ReadLevel(u32 level, memory::Span<Byte> destination) noexcept
{
    return m_inner->ReadLevel(level, destination);
}

u32 GpuTexturePool::TiledSource::PageCount(u32 firstLevel, u32 levelCount) const noexcept
{
    u32 pages = 0;
    bool packed = false;
    for (u32 level = firstLevel; level < firstLevel + levelCount; ++level)
    {
        if (level < m_tiling.standardMipCount)
        {
            pages += m_tiling.MipTileCount(level);
        }
        else
        {
            packed = true;
        }
    }
    return pages + (packed ? m_tiling.packedTileCount : 0u);
}

u32 GpuTexturePool::TiledSource::TailLevel() const noexcept
{
    // Packed mips map as a unit, so they must stream as the tail. A driver
    // that packs nothing gives every small mip a tile of its own; the
    // streamer's size rule then picks the tail, and PageCount charges a tile
    // per level.
    return m_tiling.packedMipCount > 0 ? m_tiling.standardMipCount : kNoLevel;
}

// ---------------------------------------------------------------------------

struct GpuTexturePool::Impl
{
    struct Entry
    {
        bool live = false;
        rhi::TextureHandle texture{};
        rhi::TextureDesc desc{};
        rhi::TextureTiling tiling{};
        TiledSource* source = nullptr;
    };

    explicit Impl(memory::Allocator& allocator) noexcept
        : allocator(&allocator), entries(allocator), mappings(allocator), staging(allocator)
    {
    }

    memory::Allocator* allocator = nullptr;
    rhi::Device* device = nullptr;
    GpuTexturePoolDesc desc{};
    rhi::TileHeapHandle heap{};
    rhi::BufferHandle uploads[kMaxFramesInFlight] = {};
    u64 uploadFenceValues[kMaxFramesInFlight] = {};
    rhi::FenceHandle fence{};
    u64 fenceValue = 0;
    u32 frame = 0;
    u64 uploadOffset = 0;
    rhi::CommandList list;
    bool listOpen = false;
    // Submitted lists stay alive until their fence passes: destroying a
    // command list releases it at once, under the GPU if it has not run yet.
    rhi::CommandList submitted[kMaxFramesInFlight];
    memory::DynamicArray<Entry> entries;  // indexed by streamer handle index
    memory::DynamicArray<rhi::TileMapping> mappings;
    memory::DynamicArray<Byte> staging;
    GpuTexturePoolStats stats{};

    [[nodiscard]] Entry* Find(TextureHandle handle) noexcept
    {
        return handle.index < entries.Size() && entries[handle.index].live ? &entries[handle.index]
                                                                           : nullptr;
    }

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

    // Submits what is recorded and signals the fence for this frame's buffer.
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
        // The slot's previous list (a flush earlier in this frame, or the
        // frame this buffer last served) must have finished before it dies.
        if (uploadFenceValues[frame] != 0)
        {
            (void)device->WaitFence(fence, uploadFenceValues[frame]);
        }
        submitted[frame] = std::move(list);
        uploadFenceValues[frame] = fenceValue;
        list = rhi::CommandList{};
        return ok;
    }

    // Out of upload space mid-frame: submit what there is, wait for it, and
    // start over in the same buffer.
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

    [[nodiscard]] bool MapTiles(const Entry& entry, u32 firstLevel, u32 levelCount,
                                memory::Span<const u32> pages, bool map) noexcept
    {
        mappings.Clear();
        usize next = 0;
        bool packedDone = false;
        for (u32 level = firstLevel; level < firstLevel + levelCount; ++level)
        {
            if (level < entry.tiling.standardMipCount)
            {
                const u32 tilesX = entry.tiling.mipTilesX[level];
                const u32 count = entry.tiling.MipTileCount(level);
                for (u32 tile = 0; tile < count; ++tile)
                {
                    rhi::TileMapping mapping;
                    mapping.texture = entry.texture;
                    mapping.mipLevel = level;
                    mapping.tileX = tile % tilesX;
                    mapping.tileY = tile / tilesX;
                    if (map)
                    {
                        if (next >= pages.Size())
                        {
                            return false;
                        }
                        mapping.heap = heap;
                        mapping.heapTile = pages[next++];
                    }
                    if (!mappings.TryPushBack(mapping))
                    {
                        return false;
                    }
                }
            }
            else if (!packedDone)
            {
                packedDone = true;
                for (u32 tile = 0; tile < entry.tiling.packedTileCount; ++tile)
                {
                    rhi::TileMapping mapping;
                    mapping.texture = entry.texture;
                    mapping.packed = true;
                    mapping.tileX = tile;
                    if (map)
                    {
                        if (next >= pages.Size())
                        {
                            return false;
                        }
                        mapping.heap = heap;
                        mapping.heapTile = pages[next++];
                    }
                    if (!mappings.TryPushBack(mapping))
                    {
                        return false;
                    }
                }
            }
        }
        if (mappings.Empty())
        {
            return true;
        }
        if (map)
        {
            stats.tileMappings += static_cast<u32>(mappings.Size());
        }
        else
        {
            stats.tileUnmappings += static_cast<u32>(mappings.Size());
        }
        return device
            ->UpdateTileMappings(
                desc.queue, memory::Span<const rhi::TileMapping>(mappings.Data(), mappings.Size()))
            .IsOk();
    }

    // Repacks one level into the upload buffer and records its copy with the
    // barriers around it.
    [[nodiscard]] bool CopyLevel(const Entry& entry, u32 level, const Byte* data) noexcept
    {
        const StreamingLevelInfo info = entry.source->Level(level);
        const rhi::Format format = entry.desc.format;
        const u32 blockWidth = rhi::FormatBlockWidth(format);
        const u32 blockHeight = rhi::FormatBlockHeight(format);
        const u32 blockBytes = rhi::FormatBytesPerBlock(format);
        const u32 blocksX = (info.width + blockWidth - 1) / blockWidth;
        const u32 rows = (info.height + blockHeight - 1) / blockHeight;
        const u64 rowBytes = static_cast<u64>(blocksX) * blockBytes;
        if (rowBytes * rows != info.bytes)
        {
            CG_LOG(Error, Renderer, "texture pool: level %u is not tightly packed", level);
            return false;
        }
        const u64 pitch = AlignUp(rowBytes, kUploadRowAlignment);
        const u64 imageBytes = pitch * rows;
        u64 offset = AlignUp(uploadOffset, kUploadOffsetAlignment);
        if (offset + imageBytes > desc.uploadBytesPerFrame)
        {
            if (!Flush())
            {
                return false;
            }
            offset = 0;
            if (imageBytes > desc.uploadBytesPerFrame)
            {
                CG_LOG(Error, Renderer, "texture pool: level %u exceeds the upload buffer", level);
                return false;
            }
        }
        if (!staging.TryResize(static_cast<usize>(imageBytes)))
        {
            return false;
        }
        for (u32 row = 0; row < rows; ++row)
        {
            std::memcpy(staging.Data() + row * pitch, data + row * rowBytes,
                        static_cast<usize>(rowBytes));
        }
        const rhi::BufferHandle upload = uploads[frame];
        if (!device
                 ->WriteBuffer(upload, offset,
                               memory::Span<const Byte>(staging.Data(), staging.Size()))
                 .IsOk())
        {
            return false;
        }
        uploadOffset = offset + imageBytes;

        const rhi::ResourceState before = device->TextureStateOf(entry.texture, level, 0);
        rhi::SubresourceRange range{};
        range.baseMipLevel = level;
        range.mipLevelCount = 1;
        range.layerCount = 1;
        rhi::BufferBarrier bufferBarrier{upload, device->BufferStateOf(upload),
                                         rhi::ResourceState::CopySource};
        const bool bufferReady = bufferBarrier.from == rhi::ResourceState::CopySource;
        const rhi::TextureBarrier toCopy{entry.texture, range, before,
                                         rhi::ResourceState::CopyDestination};
        if (!list.Barrier(
                     memory::Span<const rhi::BufferBarrier>(&bufferBarrier, bufferReady ? 0u : 1u),
                     memory::Span<const rhi::TextureBarrier>(&toCopy, 1))
                 .IsOk())
        {
            return false;
        }
        rhi::BufferTextureCopy copy{};
        copy.buffer = upload;
        copy.bufferOffset = offset;
        copy.bufferRowPitch = static_cast<u32>(pitch);
        copy.texture = entry.texture;
        copy.mipLevel = level;
        // Block-compressed mips smaller than a block are still a whole block in
        // memory, which is the footprint the API copies.
        copy.width = blocksX * blockWidth;
        copy.height = rows * blockHeight;
        copy.depth = 1;
        if (!list.CopyBufferToTexture(copy).IsOk())
        {
            return false;
        }
        const rhi::TextureBarrier toSample{entry.texture, range,
                                           rhi::ResourceState::CopyDestination,
                                           rhi::ResourceState::ShaderRead};
        return list
            .Barrier(memory::Span<const rhi::BufferBarrier>{},
                     memory::Span<const rhi::TextureBarrier>(&toSample, 1))
            .IsOk();
    }
};

// ---------------------------------------------------------------------------

GpuTexturePool::~GpuTexturePool()
{
    Shutdown();
}

core::Status<core::Error> GpuTexturePool::Initialize(rhi::Device& device,
                                                     const GpuTexturePoolDesc& desc,
                                                     memory::Allocator& allocator) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (m_impl != nullptr)
    {
        return StatusType::Err(
            MakeError(core::ErrorCode::AlreadyExists, "texture pool is initialized"));
    }
    if (!device.Capabilities().features.Has(rhi::Feature::ReservedTextures))
    {
        return StatusType::Err(
            MakeError(core::ErrorCode::Unsupported, "texture pool needs reserved textures"));
    }
    if (desc.tileCount == 0 || desc.framesInFlight == 0 ||
        desc.framesInFlight > kMaxFramesInFlight ||
        desc.uploadBytesPerFrame < kUploadOffsetAlignment)
    {
        return StatusType::Err(
            MakeError(core::ErrorCode::InvalidArgument, "invalid texture pool description"));
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

    auto heap = device.CreateTileHeap(rhi::TileHeapDesc{desc.tileCount, desc.label});
    if (!heap)
    {
        const core::Error error = heap.Error();
        Shutdown();
        return StatusType::Err(error);
    }
    impl.heap = heap.Value();
    for (u32 index = 0; index < desc.framesInFlight; ++index)
    {
        rhi::BufferDesc bufferDesc{};
        bufferDesc.usage = rhi::BufferUsage::HostVisible | rhi::BufferUsage::CopySource;
        bufferDesc.sizeBytes = desc.uploadBytesPerFrame;
        bufferDesc.label = "texture_pool_upload";
        auto buffer = device.CreateBuffer(bufferDesc);
        if (!buffer)
        {
            const core::Error error = buffer.Error();
            Shutdown();
            return StatusType::Err(error);
        }
        impl.uploads[index] = buffer.Value();
    }
    auto fence = device.CreateFence(rhi::FenceDesc{});
    if (!fence)
    {
        const core::Error error = fence.Error();
        Shutdown();
        return StatusType::Err(error);
    }
    impl.fence = fence.Value();
    return StatusType::Ok();
}

void GpuTexturePool::Shutdown() noexcept
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
    for (usize index = 0; index < impl.entries.Size(); ++index)
    {
        Impl::Entry& entry = impl.entries[index];
        if (entry.texture.IsValid())
        {
            (void)impl.device->DestroyTexture(entry.texture);
        }
        if (entry.source != nullptr)
        {
            entry.source->~TiledSource();
            impl.allocator->Deallocate(entry.source);
        }
    }
    for (const rhi::BufferHandle buffer : impl.uploads)
    {
        if (buffer.IsValid())
        {
            (void)impl.device->DestroyBuffer(buffer);
        }
    }
    if (impl.heap.IsValid())
    {
        (void)impl.device->DestroyTileHeap(impl.heap);
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

core::Result<TextureHandle, core::Error> GpuTexturePool::AddTexture(TextureStreamer& streamer,
                                                                    TextureMipSource& source,
                                                                    const rhi::TextureDesc& desc,
                                                                    const char* name) noexcept
{
    using ResultType = core::Result<TextureHandle, core::Error>;
    if (m_impl == nullptr)
    {
        return ResultType::Err(
            MakeError(core::ErrorCode::InvalidArgument, "pool is not initialized"));
    }
    Impl& impl = *m_impl;
    rhi::TextureDesc reserved = desc;
    reserved.reserved = true;
    reserved.usage =
        reserved.usage | rhi::TextureUsage::Sampled | rhi::TextureUsage::CopyDestination;
    reserved.label = name;
    if (reserved.mipLevels != source.LevelCount())
    {
        return ResultType::Err(MakeError(core::ErrorCode::InvalidArgument,
                                         "texture mip count differs from its source"));
    }
    auto texture = impl.device->CreateTexture(reserved);
    if (!texture)
    {
        return ResultType::Err(texture.Error());
    }
    const rhi::TextureTiling tiling = impl.device->TextureTilingOf(texture.Value());
    void* memory = impl.allocator->Allocate(sizeof(TiledSource), alignof(TiledSource));
    if (!tiling.valid || memory == nullptr)
    {
        impl.allocator->Deallocate(memory);
        (void)impl.device->DestroyTexture(texture.Value());
        return ResultType::Err(MakeError(core::ErrorCode::Internal, "texture tiling unavailable"));
    }
    auto* tiled = new (memory) TiledSource(source, tiling);
    auto handle = streamer.Register(*tiled, name);
    if (!handle)
    {
        tiled->~TiledSource();
        impl.allocator->Deallocate(tiled);
        (void)impl.device->DestroyTexture(texture.Value());
        return ResultType::Err(handle.Error());
    }
    const u32 index = handle.Value().index;
    while (impl.entries.Size() <= index)
    {
        if (!impl.entries.TryPushBack(Impl::Entry{}))
        {
            return ResultType::Err(MakeError(core::ErrorCode::OutOfRange, "out of memory"));
        }
    }
    Impl::Entry& entry = impl.entries[index];
    if (entry.source != nullptr)
    {
        entry.source->~TiledSource();
        impl.allocator->Deallocate(entry.source);
    }
    entry.live = true;
    entry.texture = texture.Value();
    entry.desc = reserved;
    entry.tiling = tiling;
    entry.source = tiled;
    return ResultType::Ok(handle.Value());
}

void GpuTexturePool::RemoveTexture(TextureStreamer& streamer, TextureHandle texture) noexcept
{
    if (m_impl == nullptr)
    {
        return;
    }
    Impl& impl = *m_impl;
    Impl::Entry* entry = impl.Find(texture);
    if (entry == nullptr)
    {
        return;
    }
    streamer.Unregister(texture, *this);
    // Anything already submitted may still sample it.
    if (impl.listOpen)
    {
        (void)impl.Submit();
        (void)impl.OpenList();
    }
    (void)impl.device->WaitFence(impl.fence, impl.fenceValue);
    (void)impl.device->DestroyTexture(entry->texture);
    entry->texture = rhi::TextureHandle{};
    entry->live = false;
    entry->source->~TiledSource();
    impl.allocator->Deallocate(entry->source);
    entry->source = nullptr;
}

core::Status<core::Error> GpuTexturePool::BeginFrame() noexcept
{
    using StatusType = core::Status<core::Error>;
    if (m_impl == nullptr)
    {
        return StatusType::Err(
            MakeError(core::ErrorCode::InvalidArgument, "pool is not initialized"));
    }
    Impl& impl = *m_impl;
    // Work recorded outside a frame (a Flush during a level load opens a list
    // on demand) is submitted with the previous buffer before moving on.
    if (impl.listOpen && !impl.Submit())
    {
        return StatusType::Err(MakeError(core::ErrorCode::Internal, "submit failed"));
    }
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
    return StatusType::Ok();
}

core::Status<core::Error> GpuTexturePool::EndFrame() noexcept
{
    using StatusType = core::Status<core::Error>;
    if (m_impl == nullptr)
    {
        return StatusType::Err(
            MakeError(core::ErrorCode::InvalidArgument, "pool is not initialized"));
    }
    return m_impl->Submit()
               ? StatusType::Ok()
               : StatusType::Err(MakeError(core::ErrorCode::Internal, "submit failed"));
}

rhi::TextureHandle GpuTexturePool::Texture(TextureHandle texture) const noexcept
{
    const Impl::Entry* entry = m_impl != nullptr ? m_impl->Find(texture) : nullptr;
    return entry != nullptr ? entry->texture : rhi::TextureHandle{};
}

rhi::TileHeapHandle GpuTexturePool::Heap() const noexcept
{
    return m_impl != nullptr ? m_impl->heap : rhi::TileHeapHandle{};
}

const GpuTexturePoolStats& GpuTexturePool::Stats() const noexcept
{
    static const GpuTexturePoolStats empty{};
    return m_impl != nullptr ? m_impl->stats : empty;
}

void GpuTexturePool::UploadLevels(TextureHandle texture, u32 firstLevel, u32 levelCount,
                                  memory::Span<const Byte> data,
                                  memory::Span<const u32> pages) noexcept
{
    if (m_impl == nullptr)
    {
        return;
    }
    Impl& impl = *m_impl;
    Impl::Entry* entry = impl.Find(texture);
    if (entry == nullptr || (!impl.listOpen && !impl.OpenList()))
    {
        return;
    }
    if (!impl.MapTiles(*entry, firstLevel, levelCount, pages, true))
    {
        CG_LOG(Error, Renderer, "texture pool: mapping %u levels of texture %u failed", levelCount,
               texture.index);
        return;
    }
    u64 offset = 0;
    for (u32 level = firstLevel; level < firstLevel + levelCount; ++level)
    {
        const u64 bytes = entry->source->Level(level).bytes;
        if (offset + bytes > data.Size() || !impl.CopyLevel(*entry, level, data.Data() + offset))
        {
            CG_LOG(Error, Renderer, "texture pool: copying level %u of texture %u failed", level,
                   texture.index);
            return;
        }
        offset += bytes;
    }
    ++impl.stats.uploads;
    impl.stats.uploadedBytes += data.Size();
}

void GpuTexturePool::EvictLevel(TextureHandle texture, u32 level,
                                memory::Span<const u32> pages) noexcept
{
    if (m_impl == nullptr)
    {
        return;
    }
    Impl& impl = *m_impl;
    Impl::Entry* entry = impl.Find(texture);
    if (entry == nullptr)
    {
        return;
    }
    // A request's pages cover one level, or the whole tail. Which it was is
    // recovered from the page count, because on a driver that packs nothing
    // the tail is several standard mips sharing one page list.
    u32 count = 1;
    while (level + count < entry->desc.mipLevels &&
           entry->source->PageCount(level, count) < pages.Size())
    {
        ++count;
    }
    if (!impl.MapTiles(*entry, level, count, memory::Span<const u32>{}, false))
    {
        CG_LOG(Error, Renderer, "texture pool: unmapping level %u of texture %u failed", level,
               texture.index);
        return;
    }
    ++impl.stats.evictions;
}

}  // namespace cyber::renderer
