#include <cyber/memory/dynamic_array.h>
#include <cyber/platform/thread.h>
#include <cyber/renderer/virtual_texture.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <new>

namespace cyber::renderer
{

using cyber::core::i32;
using cyber::core::u8;

namespace
{

constexpr u32 kNone = 0xFFFFFFFFu;
constexpr u32 kMaxMips = 16;

enum class PageState : u8
{
    Absent = 0,
    Loading = 1,
    Resident = 2,
};

core::Error MakeError(core::ErrorCode code, const char* message) noexcept
{
    return core::Error{code, message};
}

bool IsPowerOfTwo(u32 value) noexcept
{
    return value != 0 && (value & (value - 1u)) == 0;
}

}  // namespace

struct VirtualTexture::Impl
{
    struct Slot
    {
        u32 page = kNone;  // flat page index held, or kNone
        u64 lastUsed = 0;
        bool pinned = false;
    };

    enum class RequestStatus : u8
    {
        Free,
        Queued,
        Reading,
        Done,
    };

    struct Request
    {
        RequestStatus status = RequestStatus::Free;
        u32 page = kNone;
        u32 mip = 0;
        u32 x = 0;
        u32 y = 0;
        Byte* texels = nullptr;
        bool ok = false;
    };

    explicit Impl(memory::Allocator& allocator) noexcept
        : allocator(&allocator), pageTable(allocator), residentSlot(allocator), state(allocator),
          slots(allocator), freeSlots(allocator), atlas(allocator), requests(allocator),
          queue(allocator), completed(allocator), ready(allocator), threads(allocator),
          candidates(allocator), dirtyCache(allocator), lastRequested(allocator), staging(allocator)
    {
    }

    memory::Allocator* allocator = nullptr;
    VirtualTextureDesc desc{};
    VirtualPageProducer* producer = nullptr;
    u32 mipCount = 0;
    u32 pagesX[kMaxMips] = {};
    u32 pagesY[kMaxMips] = {};
    u32 pageOffset[kMaxMips + 1] = {};
    u32 stride = 0;  // texels per page side, border included

    memory::DynamicArray<u32> pageTable;      // encoded entries, flat over mips
    memory::DynamicArray<u32> residentSlot;   // flat page -> slot or kNone
    memory::DynamicArray<PageState> state;    // flat page -> state
    std::atomic<u32>* requestBits = nullptr;  // flat page bitset
    u32 requestWords = 0;
    memory::DynamicArray<Slot> slots;
    memory::DynamicArray<u32> freeSlots;
    memory::DynamicArray<Byte> atlas;

    memory::DynamicArray<Request> requests;
    memory::DynamicArray<u32> queue;      // guarded by mutex
    memory::DynamicArray<u32> completed;  // guarded by mutex
    memory::DynamicArray<u32> ready;      // main thread
    platform::Mutex mutex;
    platform::ConditionVariable workAvailable;
    platform::ConditionVariable workFinished;
    bool stopping = false;
    memory::DynamicArray<platform::Thread*> threads;

    struct Candidate
    {
        u32 page = 0;
        u32 mip = 0;
        u32 order = 0;
    };
    memory::DynamicArray<Candidate> candidates;
    memory::DynamicArray<u32> dirtyCache;
    memory::DynamicArray<u32> lastRequested;
    memory::DynamicArray<Byte> staging;
    bool dirtyMip[kMaxMips] = {};
    VirtualTextureStats stats{};
    u64 frame = 1;

    // ---- Addressing ------------------------------------------------------

    [[nodiscard]] u32 Flat(u32 mip, u32 x, u32 y) const noexcept
    {
        return pageOffset[mip] + y * pagesX[mip] + x;
    }

    void Unflatten(u32 page, u32& mip, u32& x, u32& y) const noexcept
    {
        mip = 0;
        while (mip + 1 < mipCount && page >= pageOffset[mip + 1])
        {
            ++mip;
        }
        const u32 local = page - pageOffset[mip];
        x = local % pagesX[mip];
        y = local / pagesX[mip];
    }

    [[nodiscard]] u32 SlotX(u32 slot) const noexcept
    {
        return slot % desc.cachePagesX;
    }
    [[nodiscard]] u32 SlotY(u32 slot) const noexcept
    {
        return slot / desc.cachePagesX;
    }

    [[nodiscard]] u32 EntryForSlot(u32 slot, u32 mip) const noexcept
    {
        return EncodePageTableEntry(VirtualPageEntry{SlotX(slot), SlotY(slot), mip});
    }

    // ---- Page table maintenance -----------------------------------------

    // Calls `visit(flatIndex, mip)` for the page itself and every descendant.
    template <typename Visitor>
    void ForSubtree(u32 mip, u32 x, u32 y, Visitor&& visit) noexcept
    {
        for (u32 level = 0; level <= mip; ++level)
        {
            const u32 child = mip - level;
            const u32 x0 = x << level;
            const u32 y0 = y << level;
            const u32 x1 = std::min((x + 1u) << level, pagesX[child]);
            const u32 y1 = std::min((y + 1u) << level, pagesY[child]);
            for (u32 cy = y0; cy < y1; ++cy)
            {
                for (u32 cx = x0; cx < x1; ++cx)
                {
                    visit(Flat(child, cx, cy), child);
                }
            }
        }
    }

    // A page became resident in `slot`: it and every descendant that resolved
    // to something coarser now resolve to it.
    void Install(u32 page, u32 slot) noexcept
    {
        u32 mip = 0;
        u32 x = 0;
        u32 y = 0;
        Unflatten(page, mip, x, y);
        const u32 entry = EntryForSlot(slot, mip);
        ForSubtree(mip, x, y,
                   [&](u32 index, u32 level) noexcept
                   {
                       if (DecodePageTableEntry(pageTable[index]).mip > mip)
                       {
                           pageTable[index] = entry;
                           dirtyMip[level] = true;
                       }
                   });
        residentSlot[page] = slot;
        state[page] = PageState::Resident;
        slots[slot].page = page;
        slots[slot].lastUsed = frame;
    }

    // A page left the cache: whatever resolved to it now resolves to what its
    // parent resolves to, which is always valid because the top mip is pinned.
    void Uninstall(u32 page) noexcept
    {
        u32 mip = 0;
        u32 x = 0;
        u32 y = 0;
        Unflatten(page, mip, x, y);
        const u32 old = pageTable[page];
        const u32 parent = pageTable[Flat(mip + 1, x >> 1u, y >> 1u)];
        ForSubtree(mip, x, y,
                   [&](u32 index, u32 level) noexcept
                   {
                       if (pageTable[index] == old)
                       {
                           pageTable[index] = parent;
                           dirtyMip[level] = true;
                       }
                   });
        residentSlot[page] = kNone;
        state[page] = PageState::Absent;
    }

    void CopyToAtlas(u32 slot, const Byte* texels) noexcept
    {
        const usize atlasStride = static_cast<usize>(desc.cachePagesX) * stride * 4u;
        Byte* origin = atlas.Data() + static_cast<usize>(SlotY(slot)) * stride * atlasStride +
                       static_cast<usize>(SlotX(slot)) * stride * 4u;
        for (u32 row = 0; row < stride; ++row)
        {
            std::memcpy(origin + row * atlasStride, texels + static_cast<usize>(row) * stride * 4u,
                        static_cast<usize>(stride) * 4u);
        }
        (void)dirtyCache.TryPushBack(slot);
    }

    // Least recently used slot not touched this frame and not pinned.
    [[nodiscard]] u32 Victim() const noexcept
    {
        u32 best = kNone;
        u64 oldest = frame;
        for (usize index = 0; index < slots.Size(); ++index)
        {
            const Slot& slot = slots[index];
            if (!slot.pinned && slot.page != kNone && slot.lastUsed < oldest)
            {
                oldest = slot.lastUsed;
                best = static_cast<u32>(index);
            }
        }
        return best;
    }

    // ---- IO -------------------------------------------------------------

    static void IoEntry(void* userData) noexcept
    {
        (void)platform::Thread::SetCurrentThreadPriority(platform::ThreadPriority::Low);
        static_cast<Impl*>(userData)->IoLoop();
    }

    void IoLoop() noexcept
    {
        while (true)
        {
            mutex.Lock();
            while (!stopping && queue.Empty())
            {
                workAvailable.Wait(mutex);
            }
            if (stopping)
            {
                mutex.Unlock();
                return;
            }
            // Coarsest first: those pages improve the most of the image.
            usize best = 0;
            for (usize i = 1; i < queue.Size(); ++i)
            {
                if (requests[queue[i]].mip > requests[queue[best]].mip)
                {
                    best = i;
                }
            }
            const u32 index = queue[best];
            queue[best] = queue[queue.Size() - 1];
            queue.PopBack();
            requests[index].status = RequestStatus::Reading;
            mutex.Unlock();

            Request& request = requests[index];
            const bool ok = producer->ProducePage(
                request.mip, request.x, request.y,
                memory::Span<Byte>(request.texels, static_cast<usize>(stride) * stride * 4u));

            mutex.Lock();
            request.ok = ok;
            request.status = RequestStatus::Done;
            (void)completed.TryPushBack(index);
            workFinished.NotifyAll();
            mutex.Unlock();
        }
    }

    [[nodiscard]] u32 FreeRequest() const noexcept
    {
        for (usize index = 0; index < requests.Size(); ++index)
        {
            if (requests[index].status == RequestStatus::Free)
            {
                return static_cast<u32>(index);
            }
        }
        return kNone;
    }

    [[nodiscard]] u32 InFlight() const noexcept
    {
        u32 count = 0;
        for (usize index = 0; index < requests.Size(); ++index)
        {
            count += requests[index].status != RequestStatus::Free ? 1u : 0u;
        }
        return count;
    }

    void Issue(u32 page) noexcept
    {
        const u32 index = FreeRequest();
        if (index == kNone)
        {
            return;
        }
        Request& request = requests[index];
        request.page = page;
        Unflatten(page, request.mip, request.x, request.y);
        request.ok = false;
        state[page] = PageState::Loading;
        mutex.Lock();
        request.status = RequestStatus::Queued;
        (void)queue.TryPushBack(index);
        workAvailable.NotifyOne();
        mutex.Unlock();
    }
};

// ---------------------------------------------------------------------------

VirtualTexture::~VirtualTexture()
{
    Shutdown();
}

core::Status<core::Error> VirtualTexture::Initialize(const VirtualTextureDesc& desc,
                                                     VirtualPageProducer& producer,
                                                     memory::Allocator& allocator) noexcept
{
    using StatusType = core::Status<core::Error>;
    if (m_impl != nullptr)
    {
        return StatusType::Err(MakeError(core::ErrorCode::AlreadyExists, "already initialized"));
    }
    if (!IsPowerOfTwo(desc.pageSize) || !IsPowerOfTwo(desc.width) || !IsPowerOfTwo(desc.height) ||
        desc.width < desc.pageSize || desc.height < desc.pageSize ||
        desc.border > desc.pageSize / 2 || desc.cachePagesX == 0 || desc.cachePagesY == 0 ||
        desc.cachePagesX > 1024 || desc.cachePagesY > 1024 || desc.ioThreads == 0 ||
        desc.ioThreads > 16 || desc.maxInFlight == 0 || desc.maxUploadsPerFrame == 0)
    {
        return StatusType::Err(
            MakeError(core::ErrorCode::InvalidArgument, "invalid virtual texture description"));
    }
    void* memory = allocator.Allocate(sizeof(Impl), alignof(Impl));
    if (memory == nullptr)
    {
        return StatusType::Err(MakeError(core::ErrorCode::OutOfRange, "out of memory"));
    }
    m_impl = new (memory) Impl(allocator);
    Impl& impl = *m_impl;
    impl.desc = desc;
    impl.producer = &producer;
    impl.stride = desc.pageSize + 2u * desc.border;

    // Mips down to the level where the smaller dimension is one page.
    u32 total = 0;
    for (u32 mip = 0; mip < kMaxMips; ++mip)
    {
        const u32 width = desc.width >> mip;
        const u32 height = desc.height >> mip;
        if (width < desc.pageSize || height < desc.pageSize)
        {
            break;
        }
        impl.pagesX[mip] = width / desc.pageSize;
        impl.pagesY[mip] = height / desc.pageSize;
        impl.pageOffset[mip] = total;
        total += impl.pagesX[mip] * impl.pagesY[mip];
        impl.mipCount = mip + 1;
    }
    impl.pageOffset[impl.mipCount] = total;
    const u32 top = impl.mipCount - 1;
    const u32 topPages = impl.pagesX[top] * impl.pagesY[top];
    const u32 cachePages = desc.cachePagesX * desc.cachePagesY;
    if (topPages >= cachePages)
    {
        Shutdown();
        return StatusType::Err(MakeError(core::ErrorCode::InvalidArgument,
                                         "the cache cannot hold the pinned top-level pages"));
    }

    impl.requestWords = (total + 31u) / 32u;
    void* bits =
        allocator.Allocate(sizeof(std::atomic<u32>) * impl.requestWords, alignof(std::atomic<u32>));
    const usize atlasBytes =
        static_cast<usize>(desc.cachePagesX) * desc.cachePagesY * impl.stride * impl.stride * 4u;
    const usize pageBytes = static_cast<usize>(impl.stride) * impl.stride * 4u;
    bool ok =
        bits != nullptr && impl.pageTable.TryResize(total) && impl.residentSlot.TryResize(total) &&
        impl.state.TryResize(total) && impl.slots.TryResize(cachePages) &&
        impl.freeSlots.TryReserve(cachePages) && impl.atlas.TryResize(atlasBytes) &&
        impl.requests.TryResize(desc.maxInFlight) && impl.queue.TryReserve(desc.maxInFlight) &&
        impl.completed.TryReserve(desc.maxInFlight) && impl.ready.TryReserve(desc.maxInFlight) &&
        impl.threads.TryReserve(desc.ioThreads) && impl.dirtyCache.TryReserve(cachePages) &&
        impl.staging.TryResize(pageBytes * (desc.maxInFlight + 1u));
    if (!ok)
    {
        allocator.Deallocate(bits);
        Shutdown();
        return StatusType::Err(MakeError(core::ErrorCode::OutOfRange, "out of memory"));
    }
    impl.requestBits = static_cast<std::atomic<u32>*>(bits);
    for (u32 word = 0; word < impl.requestWords; ++word)
    {
        new (&impl.requestBits[word]) std::atomic<u32>(0u);
    }
    // Every entry starts coarser than any real mip, so installing the top
    // pages below claims the whole table.
    const u32 unresolved = EncodePageTableEntry(VirtualPageEntry{0, 0, 0x1Fu});
    for (u32 page = 0; page < total; ++page)
    {
        impl.residentSlot[page] = kNone;
        impl.state[page] = PageState::Absent;
        impl.pageTable[page] = unresolved;
    }
    for (u32 index = 0; index < desc.maxInFlight; ++index)
    {
        impl.requests[index].texels = impl.staging.Data() + pageBytes * (index + 1u);
    }

    // The top level, produced here and pinned: every page-table entry starts
    // out resolving to the top page that covers it.
    Byte* scratch = impl.staging.Data();
    for (u32 y = 0; y < impl.pagesY[top]; ++y)
    {
        for (u32 x = 0; x < impl.pagesX[top]; ++x)
        {
            const u32 slot = y * impl.pagesX[top] + x;
            if (!producer.ProducePage(top, x, y, memory::Span<Byte>(scratch, pageBytes)))
            {
                Shutdown();
                return StatusType::Err(MakeError(core::ErrorCode::Internal,
                                                 "the top-level page could not be produced"));
            }
            impl.CopyToAtlas(slot, scratch);
            impl.Install(impl.Flat(top, x, y), slot);
            impl.slots[slot].pinned = true;
        }
    }
    for (u32 slot = cachePages; slot > topPages; --slot)
    {
        (void)impl.freeSlots.TryPushBack(slot - 1);
    }
    for (u32 mip = 0; mip < impl.mipCount; ++mip)
    {
        impl.dirtyMip[mip] = true;
    }

    for (u32 index = 0; index < desc.ioThreads; ++index)
    {
        void* threadMemory =
            allocator.Allocate(sizeof(platform::Thread), alignof(platform::Thread));
        if (threadMemory == nullptr)
        {
            Shutdown();
            return StatusType::Err(MakeError(core::ErrorCode::OutOfRange, "out of memory"));
        }
        auto* thread = new (threadMemory) platform::Thread();
        (void)impl.threads.TryPushBack(thread);
        if (!thread->Start(allocator, "vt_io", &Impl::IoEntry, &impl).IsOk())
        {
            Shutdown();
            return StatusType::Err(MakeError(core::ErrorCode::Internal, "IO thread start failed"));
        }
    }
    impl.stats.mipCount = impl.mipCount;
    impl.stats.virtualPages = total;
    impl.stats.cachePages = cachePages;
    impl.stats.residentPages = topPages;
    return StatusType::Ok();
}

void VirtualTexture::Shutdown() noexcept
{
    if (m_impl == nullptr)
    {
        return;
    }
    Impl& impl = *m_impl;
    impl.mutex.Lock();
    impl.stopping = true;
    impl.workAvailable.NotifyAll();
    impl.mutex.Unlock();
    for (usize index = 0; index < impl.threads.Size(); ++index)
    {
        platform::Thread* thread = impl.threads[index];
        if (thread->Joinable())
        {
            thread->Join();
        }
        thread->~Thread();
        impl.allocator->Deallocate(thread);
    }
    if (impl.requestBits != nullptr)
    {
        impl.allocator->Deallocate(impl.requestBits);
    }
    memory::Allocator* allocator = impl.allocator;
    impl.~Impl();
    allocator->Deallocate(m_impl);
    m_impl = nullptr;
}

u32 VirtualTexture::MipCount() const noexcept
{
    return m_impl != nullptr ? m_impl->mipCount : 0u;
}

u32 VirtualTexture::PagesX(u32 mip) const noexcept
{
    return m_impl != nullptr && mip < m_impl->mipCount ? m_impl->pagesX[mip] : 0u;
}

u32 VirtualTexture::PagesY(u32 mip) const noexcept
{
    return m_impl != nullptr && mip < m_impl->mipCount ? m_impl->pagesY[mip] : 0u;
}

void VirtualTexture::RequestPage(u32 mip, u32 pageX, u32 pageY) noexcept
{
    if (m_impl == nullptr || mip >= m_impl->mipCount || pageX >= m_impl->pagesX[mip] ||
        pageY >= m_impl->pagesY[mip])
    {
        return;
    }
    const u32 page = m_impl->Flat(mip, pageX, pageY);
    std::atomic<u32>& word = m_impl->requestBits[page >> 5u];
    const u32 bit = 1u << (page & 31u);
    // Most samples of a frame hit pages already requested; a relaxed load
    // keeps them off the cache line's write path.
    if ((word.load(std::memory_order_relaxed) & bit) == 0)
    {
        word.fetch_or(bit, std::memory_order_relaxed);
    }
}

void VirtualTexture::RequestSample(f32 u, f32 v, f32 lod) noexcept
{
    if (m_impl == nullptr)
    {
        return;
    }
    const Impl& impl = *m_impl;
    const f32 level = std::isfinite(lod) ? std::max(lod, 0.0f) : 0.0f;
    const u32 mip = std::min(static_cast<u32>(level), impl.mipCount - 1u);
    const f32 uu = u - std::floor(u);
    const f32 vv = v - std::floor(v);
    const u32 x =
        std::min(static_cast<u32>(uu * static_cast<f32>(impl.pagesX[mip])), impl.pagesX[mip] - 1u);
    const u32 y =
        std::min(static_cast<u32>(vv * static_cast<f32>(impl.pagesY[mip])), impl.pagesY[mip] - 1u);
    RequestPage(mip, x, y);
}

VirtualTextureStats VirtualTexture::Update() noexcept
{
    if (m_impl == nullptr)
    {
        return VirtualTextureStats{};
    }
    Impl& impl = *m_impl;
    ++impl.frame;
    VirtualTextureStats& stats = impl.stats;
    stats.requestedThisFrame = 0;
    stats.missingThisFrame = 0;
    stats.uploadsThisFrame = 0;
    stats.evictionsThisFrame = 0;
    stats.producerErrorsThisFrame = 0;
    stats.cacheFullThisFrame = 0;
    impl.dirtyCache.Clear();
    for (u32 mip = 0; mip < impl.mipCount; ++mip)
    {
        impl.dirtyMip[mip] = false;
    }

    // 1. The frame's feedback: touch what is resident (and what the misses
    //    fall back to, so the fallback survives), and collect the next level
    //    of refinement for each miss: the coarsest page on its path that is
    //    not yet resident.
    impl.candidates.Clear();
    impl.lastRequested.Clear();
    u32 order = 0;
    for (u32 word = 0; word < impl.requestWords; ++word)
    {
        u32 bits = impl.requestBits[word].exchange(0u, std::memory_order_relaxed);
        while (bits != 0)
        {
            const u32 bit = static_cast<u32>(std::countr_zero(bits));
            bits &= bits - 1u;
            const u32 page = word * 32u + bit;
            ++stats.requestedThisFrame;
            (void)impl.lastRequested.TryPushBack(page);
            const u32 resolved = impl.pageTable[page];
            const VirtualPageEntry entry = DecodePageTableEntry(resolved);
            impl.slots[entry.cacheY * impl.desc.cachePagesX + entry.cacheX].lastUsed = impl.frame;
            if (impl.state[page] == PageState::Resident)
            {
                continue;
            }
            ++stats.missingThisFrame;
            u32 mip = 0;
            u32 x = 0;
            u32 y = 0;
            impl.Unflatten(page, mip, x, y);
            // Walk up to the first resident ancestor; the page just below it
            // on this path is what to load next.
            u32 next = page;
            u32 nextMip = mip;
            while (nextMip + 1 < impl.mipCount)
            {
                const u32 parent =
                    impl.Flat(nextMip + 1, x >> (nextMip + 1 - mip), y >> (nextMip + 1 - mip));
                if (impl.state[parent] == PageState::Resident)
                {
                    break;
                }
                next = parent;
                ++nextMip;
            }
            if (impl.state[next] == PageState::Absent)
            {
                (void)impl.candidates.TryPushBack(Impl::Candidate{next, nextMip, order++});
            }
        }
    }

    // 2. Finished pages into the cache, coarsest first, within the frame's
    //    upload budget.
    impl.mutex.Lock();
    for (usize index = 0; index < impl.completed.Size(); ++index)
    {
        (void)impl.ready.TryPushBack(impl.completed[index]);
    }
    impl.completed.Clear();
    impl.mutex.Unlock();
    std::sort(impl.ready.Data(), impl.ready.Data() + impl.ready.Size(),
              [&impl](u32 lhs, u32 rhs) noexcept
              {
                  return impl.requests[lhs].mip > impl.requests[rhs].mip ||
                         (impl.requests[lhs].mip == impl.requests[rhs].mip && lhs < rhs);
              });
    usize applied = 0;
    while (applied < impl.ready.Size() && stats.uploadsThisFrame < impl.desc.maxUploadsPerFrame)
    {
        Impl::Request& request = impl.requests[impl.ready[applied]];
        if (!request.ok)
        {
            ++stats.producerErrorsThisFrame;
            impl.state[request.page] = PageState::Absent;
            request.page = kNone;
            request.ok = false;
            request.status = Impl::RequestStatus::Free;
            ++applied;
            continue;
        }
        u32 slot = kNone;
        if (!impl.freeSlots.Empty())
        {
            slot = impl.freeSlots[impl.freeSlots.Size() - 1];
            impl.freeSlots.PopBack();
        }
        else
        {
            slot = impl.Victim();
            if (slot == kNone)
            {
                // Every cache page was used this frame: the working set is
                // larger than the cache. Keep the page for a later frame.
                ++stats.cacheFullThisFrame;
                break;
            }
            impl.Uninstall(impl.slots[slot].page);
            impl.slots[slot].page = kNone;
            ++stats.evictionsThisFrame;
            ++stats.totalEvictions;
        }
        impl.CopyToAtlas(slot, request.texels);
        impl.Install(request.page, slot);
        request.status = Impl::RequestStatus::Free;
        request.page = kNone;
        ++stats.uploadsThisFrame;
        ++stats.totalUploads;
        ++applied;
    }
    usize write = 0;
    for (usize index = applied; index < impl.ready.Size(); ++index)
    {
        impl.ready[write++] = impl.ready[index];
    }
    while (impl.ready.Size() > write)
    {
        impl.ready.PopBack();
    }

    // 3. New loads, coarsest first.
    std::sort(impl.candidates.Data(), impl.candidates.Data() + impl.candidates.Size(),
              [](const Impl::Candidate& lhs, const Impl::Candidate& rhs) noexcept
              { return lhs.mip > rhs.mip || (lhs.mip == rhs.mip && lhs.order < rhs.order); });
    for (usize index = 0; index < impl.candidates.Size(); ++index)
    {
        const u32 page = impl.candidates[index].page;
        if (impl.state[page] != PageState::Absent)
        {
            continue;
        }
        if (impl.FreeRequest() == kNone)
        {
            break;
        }
        impl.Issue(page);
    }

    u32 resident = 0;
    for (usize index = 0; index < impl.slots.Size(); ++index)
    {
        resident += impl.slots[index].page != kNone ? 1u : 0u;
    }
    stats.residentPages = resident;
    stats.inFlight = impl.InFlight();
    return stats;
}

void VirtualTexture::Flush() noexcept
{
    if (m_impl == nullptr)
    {
        return;
    }
    Impl& impl = *m_impl;
    memory::DynamicArray<u32> wanted(*impl.allocator);
    for (u32 iteration = 0; iteration < 100000; ++iteration)
    {
        if (iteration > 0)
        {
            for (usize index = 0; index < wanted.Size(); ++index)
            {
                u32 mip = 0;
                u32 x = 0;
                u32 y = 0;
                impl.Unflatten(wanted[index], mip, x, y);
                RequestPage(mip, x, y);
            }
        }
        const VirtualTextureStats stats = Update();
        if (iteration == 0)
        {
            for (usize index = 0; index < impl.lastRequested.Size(); ++index)
            {
                (void)wanted.TryPushBack(impl.lastRequested[index]);
            }
        }
        if (stats.missingThisFrame == 0 && stats.inFlight == 0)
        {
            return;
        }
        if (stats.cacheFullThisFrame > 0 && stats.uploadsThisFrame == 0)
        {
            return;  // the request set does not fit the cache
        }
        if (stats.inFlight > 0 && impl.ready.Empty())
        {
            impl.mutex.Lock();
            while (impl.completed.Empty())
            {
                impl.workFinished.Wait(impl.mutex);
            }
            impl.mutex.Unlock();
        }
    }
}

VirtualPageEntry VirtualTexture::Resolve(f32 u, f32 v, f32 lod) const noexcept
{
    if (m_impl == nullptr)
    {
        return VirtualPageEntry{};
    }
    const Impl& impl = *m_impl;
    const f32 level = std::isfinite(lod) ? std::max(lod, 0.0f) : 0.0f;
    const u32 mip = std::min(static_cast<u32>(level), impl.mipCount - 1u);
    const f32 uu = u - std::floor(u);
    const f32 vv = v - std::floor(v);
    const u32 x =
        std::min(static_cast<u32>(uu * static_cast<f32>(impl.pagesX[mip])), impl.pagesX[mip] - 1u);
    const u32 y =
        std::min(static_cast<u32>(vv * static_cast<f32>(impl.pagesY[mip])), impl.pagesY[mip] - 1u);
    return DecodePageTableEntry(impl.pageTable[impl.Flat(mip, x, y)]);
}

void VirtualTexture::Sample(f32 u, f32 v, f32 lod, f32 rgba[4]) const noexcept
{
    rgba[0] = rgba[1] = rgba[2] = 0.0f;
    rgba[3] = 1.0f;
    if (m_impl == nullptr)
    {
        return;
    }
    const Impl& impl = *m_impl;
    const VirtualPageEntry entry = Resolve(u, v, lod);
    const f32 uu = u - std::floor(u);
    const f32 vv = v - std::floor(v);
    // Texel position in the resident mip, then inside its page.
    const f32 page = static_cast<f32>(impl.desc.pageSize);
    const f32 tu = uu * static_cast<f32>(impl.desc.width >> entry.mip);
    const f32 tv = vv * static_cast<f32>(impl.desc.height >> entry.mip);
    const f32 inU = tu - std::floor(tu / page) * page;
    const f32 inV = tv - std::floor(tv / page) * page;
    // Bilinear about texel centres; the border keeps all four taps inside the
    // page's own texels.
    const f32 border = static_cast<f32>(impl.desc.border);
    const f32 ax = static_cast<f32>(entry.cacheX * impl.stride) + border + inU - 0.5f;
    const f32 ay = static_cast<f32>(entry.cacheY * impl.stride) + border + inV - 0.5f;
    const f32 fx = std::floor(ax);
    const f32 fy = std::floor(ay);
    const f32 wx = ax - fx;
    const f32 wy = ay - fy;
    const i32 x0 = static_cast<i32>(fx);
    const i32 y0 = static_cast<i32>(fy);
    const usize atlasStride = static_cast<usize>(impl.desc.cachePagesX) * impl.stride * 4u;
    const auto texel = [&](i32 x, i32 y, u32 channel) noexcept
    {
        return static_cast<f32>(static_cast<u8>(impl.atlas[static_cast<usize>(y) * atlasStride +
                                                           static_cast<usize>(x) * 4u + channel])) *
               (1.0f / 255.0f);
    };
    for (u32 channel = 0; channel < 4; ++channel)
    {
        const f32 top = texel(x0, y0, channel) * (1.0f - wx) + texel(x0 + 1, y0, channel) * wx;
        const f32 bottom =
            texel(x0, y0 + 1, channel) * (1.0f - wx) + texel(x0 + 1, y0 + 1, channel) * wx;
        rgba[channel] = top * (1.0f - wy) + bottom * wy;
    }
}

memory::Span<const u32> VirtualTexture::PageTable(u32 mip) const noexcept
{
    if (m_impl == nullptr || mip >= m_impl->mipCount)
    {
        return {};
    }
    return memory::Span<const u32>(m_impl->pageTable.Data() + m_impl->pageOffset[mip],
                                   static_cast<usize>(m_impl->pagesX[mip]) * m_impl->pagesY[mip]);
}

memory::Span<const Byte> VirtualTexture::Atlas() const noexcept
{
    return m_impl != nullptr ? memory::Span<const Byte>(m_impl->atlas.Data(), m_impl->atlas.Size())
                             : memory::Span<const Byte>{};
}

u32 VirtualTexture::AtlasWidth() const noexcept
{
    return m_impl != nullptr ? m_impl->desc.cachePagesX * m_impl->stride : 0u;
}

u32 VirtualTexture::AtlasHeight() const noexcept
{
    return m_impl != nullptr ? m_impl->desc.cachePagesY * m_impl->stride : 0u;
}

memory::Span<const u32> VirtualTexture::DirtyCachePages() const noexcept
{
    return m_impl != nullptr
               ? memory::Span<const u32>(m_impl->dirtyCache.Data(), m_impl->dirtyCache.Size())
               : memory::Span<const u32>{};
}

bool VirtualTexture::PageTableDirty(u32 mip) const noexcept
{
    return m_impl != nullptr && mip < m_impl->mipCount && m_impl->dirtyMip[mip];
}

const VirtualTextureStats& VirtualTexture::Stats() const noexcept
{
    static const VirtualTextureStats empty{};
    return m_impl != nullptr ? m_impl->stats : empty;
}

}  // namespace cyber::renderer
