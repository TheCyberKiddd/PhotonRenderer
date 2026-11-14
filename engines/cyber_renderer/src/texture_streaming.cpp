#include <cyber/platform/thread.h>
#include <cyber/renderer/texture_streaming.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

namespace cyber::renderer
{

using cyber::core::i32;

namespace
{

enum class LevelState : u32
{
    Absent = 0,
    Loading = 1,
    Resident = 2,
};

core::Error MakeError(core::ErrorCode code, const char* message) noexcept
{
    return core::Error{code, message};
}

}  // namespace

f32 ComputeWantedMip(u32 textureWidth, u32 textureHeight, f32 screenPixels, f32 uvScale) noexcept
{
    const f32 texels = static_cast<f32>(std::max(textureWidth, textureHeight)) *
                       (uvScale > 0.0f && std::isfinite(uvScale) ? uvScale : 1.0f);
    if (!(screenPixels > 0.0f) || !std::isfinite(screenPixels))
    {
        // Off screen or degenerate: only the coarsest level is useful.
        return static_cast<f32>(kMaxStreamedLevels);
    }
    const f32 ratio = texels / screenPixels;
    return ratio <= 1.0f ? 0.0f : std::log2(ratio);
}

// ---------------------------------------------------------------------------

struct TextureStreamer::Impl
{
    struct Texture
    {
        TextureMipSource* source = nullptr;
        char name[48] = {};
        u32 generation = 0;
        bool live = false;
        u32 levelCount = 0;
        u32 tailLevel = 0;
        StreamingLevelInfo levels[kMaxStreamedLevels] = {};
        LevelState state[kMaxStreamedLevels] = {};
        // Pages per level; the tail's pages are stored on `tailLevel`.
        memory::DynamicArray<u32> pages[kMaxStreamedLevels];
        u32 residentLevel = 0;
        u32 wantedLevel = 0;
        u32 loadingLevel = kNoLevel;
        f32 coverage = 0.0f;
        f32 reportedMip = 0.0f;
        u64 lastUsedFrame = 0;
        bool reportedThisFrame = false;
        u32 inFlight = 0;

        explicit Texture(memory::Allocator& allocator) noexcept
            : pages{memory::DynamicArray<u32>(allocator), memory::DynamicArray<u32>(allocator),
                    memory::DynamicArray<u32>(allocator), memory::DynamicArray<u32>(allocator),
                    memory::DynamicArray<u32>(allocator), memory::DynamicArray<u32>(allocator),
                    memory::DynamicArray<u32>(allocator), memory::DynamicArray<u32>(allocator),
                    memory::DynamicArray<u32>(allocator), memory::DynamicArray<u32>(allocator),
                    memory::DynamicArray<u32>(allocator), memory::DynamicArray<u32>(allocator),
                    memory::DynamicArray<u32>(allocator), memory::DynamicArray<u32>(allocator),
                    memory::DynamicArray<u32>(allocator), memory::DynamicArray<u32>(allocator)}
        {
        }

        [[nodiscard]] u64 BytesFrom(u32 first, u32 last) const noexcept
        {
            u64 total = 0;
            for (u32 level = first; level < last; ++level)
            {
                total += levels[level].bytes;
            }
            return total;
        }
        [[nodiscard]] bool TailResident() const noexcept
        {
            return state[tailLevel] == LevelState::Resident;
        }
    };

    enum class RequestStatus : u32
    {
        Free,
        Queued,
        Reading,
        Done,
    };

    struct Request
    {
        RequestStatus status = RequestStatus::Free;
        u32 texture = 0;
        u32 generation = 0;
        TextureMipSource* source = nullptr;
        u32 firstLevel = 0;
        u32 levelCount = 0;
        Byte* staging = nullptr;
        u64 stagingCapacity = 0;
        u64 bytes = 0;
        u32 pages = 0;
        u64 levelBytes[kMaxStreamedLevels] = {};
        bool ok = false;
        f32 priority = 0.0f;
    };

    explicit Impl(memory::Allocator& allocator) noexcept
        : allocator(&allocator), textures(allocator), freePages(allocator), requests(allocator),
          queue(allocator), completed(allocator), ready(allocator), threads(allocator),
          stagingPool(allocator), scratchCandidates(allocator), scratchVictims(allocator),
          scratchPlan(allocator)
    {
    }

    memory::Allocator* allocator = nullptr;
    TextureStreamerDesc desc{};
    memory::DynamicArray<Texture*> textures;
    memory::DynamicArray<u32> freePages;
    u32 totalPages = 0;

    // Requests: a fixed pool indexed by the queues below.
    memory::DynamicArray<Request> requests;
    memory::DynamicArray<u32> queue;      // guarded by mutex: waiting for an IO thread
    memory::DynamicArray<u32> completed;  // guarded by mutex: read, waiting for Update
    memory::DynamicArray<u32> ready;      // main thread: read, waiting for upload budget
    platform::Mutex mutex;
    platform::ConditionVariable workAvailable;
    platform::ConditionVariable workFinished;
    bool stopping = false;
    memory::DynamicArray<platform::Thread*> threads;

    StreamingStats stats{};
    u64 frame = 0;
    u64 stagingBytes = 0;

    struct StagingBlock
    {
        Byte* data = nullptr;
        u64 capacity = 0;
    };
    memory::DynamicArray<StagingBlock> stagingPool;
    u64 stagingPoolBytes = 0;

    struct Candidate
    {
        u32 texture = 0;
        f32 priority = 0.0f;
    };
    memory::DynamicArray<Candidate> scratchCandidates;
    memory::DynamicArray<Candidate> scratchVictims;
    struct EvictionStep
    {
        u32 texture = 0;
        u32 levels = 0;
    };
    memory::DynamicArray<EvictionStep> scratchPlan;

    // ---- IO thread ----------------------------------------------------------

    static void IoThreadEntry(void* userData) noexcept
    {
        // Streaming must not take cores from the frame: reads wait on storage
        // anyway, and a below-normal thread yields to the main and job threads.
        (void)platform::Thread::SetCurrentThreadPriority(platform::ThreadPriority::Low);
        static_cast<Impl*>(userData)->IoLoop();
    }

    void IoLoop() noexcept
    {
        while (true)
        {
            u32 index = 0;
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
                // Highest priority first; the queue is short.
                usize best = 0;
                for (usize i = 1; i < queue.Size(); ++i)
                {
                    if (requests[queue[i]].priority > requests[queue[best]].priority)
                    {
                        best = i;
                    }
                }
                index = queue[best];
                queue[best] = queue[queue.Size() - 1];
                queue.PopBack();
                requests[index].status = RequestStatus::Reading;
                mutex.Unlock();
            }

            Request& request = requests[index];
            bool ok = true;
            u64 offset = 0;
            for (u32 i = 0; i < request.levelCount && ok; ++i)
            {
                const u64 size = request.levelBytes[i];
                ok = request.source->ReadLevel(
                    request.firstLevel + i,
                    memory::Span<Byte>(request.staging + offset, static_cast<usize>(size)));
                offset += size;
            }

            mutex.Lock();
            request.ok = ok;
            request.status = RequestStatus::Done;
            (void)completed.TryPushBack(index);  // capacity reserved at init
            workFinished.NotifyAll();
            mutex.Unlock();
        }
    }

    // ---- Pages ----------------------------------------------------------------

    [[nodiscard]] u32 PagesFor(u64 bytes) const noexcept
    {
        return static_cast<u32>((bytes + desc.pageBytes - 1) / desc.pageBytes);
    }

    // Pages a read of levels first..last-1 needs: the source's own count (a
    // reserved GPU texture's tiles) when it has one, else the bytes in pages.
    [[nodiscard]] u32 RequestPages(const Texture& texture, u32 first, u32 last) const noexcept
    {
        const u32 reported =
            texture.source != nullptr ? texture.source->PageCount(first, last - first) : 0u;
        return reported != 0 ? reported : PagesFor(texture.BytesFrom(first, last));
    }

    [[nodiscard]] bool AllocatePages(u32 count, memory::DynamicArray<u32>& out) noexcept
    {
        if (count > freePages.Size() || !out.TryReserve(count))
        {
            return false;
        }
        out.Clear();
        for (u32 i = 0; i < count; ++i)
        {
            (void)out.TryPushBack(freePages[freePages.Size() - 1]);
            freePages.PopBack();
        }
        return true;
    }

    void FreePages(memory::DynamicArray<u32>& pages) noexcept
    {
        for (usize i = 0; i < pages.Size(); ++i)
        {
            (void)freePages.TryPushBack(pages[i]);  // capacity == totalPages
        }
        pages.Clear();
    }

    // ---- Helpers --------------------------------------------------------------

    [[nodiscard]] Texture* Resolve(TextureHandle handle) const noexcept
    {
        if (handle.index >= textures.Size())
        {
            return nullptr;
        }
        Texture* texture = textures[handle.index];
        return texture != nullptr && texture->live && texture->generation == handle.generation
                   ? texture
                   : nullptr;
    }

    // ReportUsage stamps the frame the next Update completes, so a report made
    // between updates is one frame ahead of `frame`; that counts as current.
    [[nodiscard]] u64 FramesSinceUse(const Texture& texture) const noexcept
    {
        return frame >= texture.lastUsedFrame ? frame - texture.lastUsedFrame : 0;
    }

    [[nodiscard]] bool IsUsed(const Texture& texture) const noexcept
    {
        return texture.lastUsedFrame != 0 && FramesSinceUse(texture) <= desc.unusedFrameLimit;
    }

    // How much a texture matters on screen, independent of what it has loaded.
    // Eviction compares this, never Priority: a priority that grows with the
    // deficit would let two equally important textures evict each other in
    // turn, each one's loss raising its claim on the other's memory.
    [[nodiscard]] f32 Importance(const Texture& texture) const noexcept
    {
        if (!IsUsed(texture))
        {
            return -static_cast<f32>(FramesSinceUse(texture));
        }
        return 1.0f + 1000.0f * texture.coverage;
    }

    // Request order: importance, scaled by how many levels are missing, so a
    // texture far from what it needs outranks one a level short.
    [[nodiscard]] f32 Priority(const Texture& texture) const noexcept
    {
        const f32 importance = Importance(texture);
        if (importance < 0.0f)
        {
            return importance;
        }
        const f32 deficit =
            texture.residentLevel > texture.wantedLevel
                ? static_cast<f32>(std::min(texture.residentLevel, texture.levelCount) -
                                   std::min(texture.wantedLevel, texture.levelCount))
                : 0.0f;
        return importance * (1.0f + deficit);
    }

    [[nodiscard]] i32 AcquireRequest() noexcept
    {
        for (usize i = 0; i < requests.Size(); ++i)
        {
            if (requests[i].status == RequestStatus::Free)
            {
                return static_cast<i32>(i);
            }
        }
        return -1;
    }

    // Issues a read of `firstLevel..lastLevel-1`. Pages and staging are taken
    // now, on the main thread, so the IO thread never allocates.
    bool Issue(u32 textureIndex, u32 firstLevel, u32 lastLevel, f32 priority) noexcept
    {
        Texture& texture = *textures[textureIndex];
        const i32 slot = AcquireRequest();
        if (slot < 0)
        {
            return false;
        }
        const u64 bytes = texture.BytesFrom(firstLevel, lastLevel);
        const u32 pageCount = RequestPages(texture, firstLevel, lastLevel);
        if (bytes == 0 || !AllocatePages(pageCount, texture.pages[firstLevel]))
        {
            return false;
        }
        u64 stagingCapacity = 0;
        Byte* staging = AcquireStaging(bytes, stagingCapacity);
        if (staging == nullptr)
        {
            FreePages(texture.pages[firstLevel]);
            return false;
        }
        Request& request = requests[static_cast<usize>(slot)];
        request.texture = textureIndex;
        request.generation = texture.generation;
        request.source = texture.source;
        request.firstLevel = firstLevel;
        request.levelCount = lastLevel - firstLevel;
        request.staging = staging;
        request.stagingCapacity = stagingCapacity;
        request.bytes = bytes;
        request.pages = pageCount;
        for (u32 level = firstLevel; level < lastLevel; ++level)
        {
            request.levelBytes[level - firstLevel] = texture.levels[level].bytes;
            texture.state[level] = LevelState::Loading;
        }
        request.ok = false;
        request.priority = priority;
        texture.loadingLevel = firstLevel;
        ++texture.inFlight;
        stagingBytes += bytes;
        ++stats.requestsThisFrame;

        mutex.Lock();
        request.status = RequestStatus::Queued;
        (void)queue.TryPushBack(static_cast<u32>(slot));  // capacity reserved at init
        workAvailable.NotifyOne();
        mutex.Unlock();
        return true;
    }

    void ReleaseRequest(Request& request) noexcept
    {
        ReleaseStaging(request.staging, request.stagingCapacity);
        stagingBytes -= request.bytes;
        request = Request{};
    }

    // Staging blocks are recycled in power-of-two size classes. Allocating
    // and freeing a multi-megabyte block per request cost the main thread a
    // VirtualAlloc/VirtualFree pair and fresh page faults on every top-mip
    // load; a recycled block is already committed and warm.
    static u64 StagingClass(u64 bytes) noexcept
    {
        u64 capacity = 64ull << 10u;
        while (capacity < bytes)
        {
            capacity <<= 1u;
        }
        return capacity;
    }

    Byte* AcquireStaging(u64 bytes, u64& capacity) noexcept
    {
        capacity = StagingClass(bytes);
        for (usize i = 0; i < stagingPool.Size(); ++i)
        {
            if (stagingPool[i].capacity == capacity)
            {
                Byte* data = stagingPool[i].data;
                stagingPoolBytes -= capacity;
                stagingPool[i] = stagingPool[stagingPool.Size() - 1];
                stagingPool.PopBack();
                return data;
            }
        }
        return static_cast<Byte*>(allocator->Allocate(static_cast<usize>(capacity), 64));
    }

    void ReleaseStaging(Byte* data, u64 capacity) noexcept
    {
        if (data == nullptr)
        {
            return;
        }
        // Keep at most what a full set of in-flight requests of this size
        // would need; beyond that, return memory to the system.
        const u64 limit = static_cast<u64>(desc.maxInFlight) * capacity;
        if (stagingPoolBytes + capacity <= limit &&
            stagingPool.TryPushBack(StagingBlock{data, capacity}))
        {
            stagingPoolBytes += capacity;
            return;
        }
        allocator->Deallocate(data);
    }

    // Applies one finished read: uploads it, or rolls it back when the
    // texture died, the read failed, or it lost its pages.
    void Apply(u32 index, TextureUploadSink& sink) noexcept
    {
        Request& request = requests[index];
        Texture* texture = request.texture < textures.Size() ? textures[request.texture] : nullptr;
        const bool alive =
            texture != nullptr && texture->live && texture->generation == request.generation;
        if (texture != nullptr && texture->generation == request.generation)
        {
            --texture->inFlight;
            texture->loadingLevel = kNoLevel;
        }
        if (!alive)
        {
            if (texture != nullptr && texture->generation == request.generation)
            {
                FreePages(texture->pages[request.firstLevel]);
            }
            ReleaseRequest(request);
            return;
        }
        if (!request.ok)
        {
            ++stats.ioErrorsThisFrame;
            for (u32 i = 0; i < request.levelCount; ++i)
            {
                texture->state[request.firstLevel + i] = LevelState::Absent;
            }
            FreePages(texture->pages[request.firstLevel]);
            ReleaseRequest(request);
            return;
        }

        const memory::DynamicArray<u32>& pages = texture->pages[request.firstLevel];
        sink.UploadLevels(
            TextureHandle{request.texture, request.generation}, request.firstLevel,
            request.levelCount,
            memory::Span<const Byte>(request.staging, static_cast<usize>(request.bytes)),
            memory::Span<const u32>(pages.Data(), pages.Size()));
        for (u32 i = 0; i < request.levelCount; ++i)
        {
            texture->state[request.firstLevel + i] = LevelState::Resident;
        }
        // The chain stays contiguous because only the level just above the
        // resident chain (or the tail) is ever requested.
        while (texture->residentLevel > 0 &&
               texture->state[texture->residentLevel - 1] == LevelState::Resident)
        {
            --texture->residentLevel;
        }
        if (request.firstLevel == texture->tailLevel)
        {
            texture->residentLevel = std::min(texture->residentLevel, texture->tailLevel);
            while (texture->residentLevel > 0 &&
                   texture->state[texture->residentLevel - 1] == LevelState::Resident)
            {
                --texture->residentLevel;
            }
        }
        ++stats.uploadsThisFrame;
        stats.uploadBytesThisFrame += request.bytes;
        ++stats.totalUploads;
        stats.totalUploadBytes += request.bytes;
        ReleaseRequest(request);
    }

    // Evicts the finest resident level of a texture. The tail is never evicted.
    bool EvictFinest(u32 textureIndex, TextureUploadSink& sink) noexcept
    {
        Texture& texture = *textures[textureIndex];
        const u32 level = texture.residentLevel;
        if (level >= texture.tailLevel || texture.state[level] != LevelState::Resident)
        {
            return false;
        }
        const memory::DynamicArray<u32>& pages = texture.pages[level];
        sink.EvictLevel(TextureHandle{textureIndex, texture.generation}, level,
                        memory::Span<const u32>(pages.Data(), pages.Size()));
        FreePages(texture.pages[level]);
        texture.state[level] = LevelState::Absent;
        ++texture.residentLevel;
        ++stats.evictionsThisFrame;
        ++stats.totalEvictions;
        return true;
    }

    void Schedule(TextureUploadSink& sink) noexcept
    {
        // Tails first: a texture without one cannot be drawn at all.
        for (usize i = 0; i < textures.Size(); ++i)
        {
            Texture* texture = textures[i];
            if (texture != nullptr && texture->live &&
                texture->state[texture->tailLevel] == LevelState::Absent && texture->inFlight == 0)
            {
                if (!Issue(static_cast<u32>(i), texture->tailLevel, texture->levelCount, 1.0e30f))
                {
                    return;
                }
            }
        }

        scratchCandidates.Clear();
        for (usize i = 0; i < textures.Size(); ++i)
        {
            Texture* texture = textures[i];
            if (texture == nullptr || !texture->live || !texture->TailResident() ||
                texture->inFlight != 0 || texture->residentLevel <= texture->wantedLevel)
            {
                continue;
            }
            (void)scratchCandidates.TryPushBack(Candidate{static_cast<u32>(i), Priority(*texture)});
        }
        std::sort(scratchCandidates.Data(), scratchCandidates.Data() + scratchCandidates.Size(),
                  [](const Candidate& lhs, const Candidate& rhs) noexcept
                  {
                      return lhs.priority > rhs.priority ||
                             (lhs.priority == rhs.priority && lhs.texture < rhs.texture);
                  });

        bool victimsBuilt = false;
        for (usize c = 0; c < scratchCandidates.Size(); ++c)
        {
            if (AcquireRequest() < 0)
            {
                break;
            }
            const Candidate candidate = scratchCandidates[c];
            Texture& texture = *textures[candidate.texture];
            const u32 level = texture.residentLevel - 1;
            const u32 needed = RequestPages(texture, level, level + 1);

            if (freePages.Size() < needed)
            {
                if (!victimsBuilt)
                {
                    BuildVictims();
                    victimsBuilt = true;
                }
                if (!PlanEvictions(candidate.texture, needed))
                {
                    // Evicting only part of what is needed would free memory
                    // the request still cannot use, and the victims would
                    // load it straight back: evict nothing instead.
                    ++stats.budgetBlockedThisFrame;
                    continue;
                }
                for (usize step = 0; step < scratchPlan.Size(); ++step)
                {
                    for (u32 count = 0; count < scratchPlan[step].levels; ++count)
                    {
                        (void)EvictFinest(scratchPlan[step].texture, sink);
                    }
                }
            }
            if (!Issue(candidate.texture, level, level + 1, candidate.priority))
            {
                break;
            }
        }
    }

    // Fills scratchPlan with the evictions that free `needed` pages for a
    // request of `requester`, or returns false when no plan can. Victims are
    // taken in BuildVictims order, finest level first. Surplus levels (finer
    // than their texture needs) are always eligible. A needed level goes only
    // to a request clearly more important than its owner, with a margin so
    // near-equal textures settle instead of trading memory. A texture never
    // evicts its own chain, and a texture with a read in flight is skipped so
    // its chain stays contiguous.
    bool PlanEvictions(u32 requester, u32 needed) noexcept
    {
        scratchPlan.Clear();
        const f32 requesterImportance = Importance(*textures[requester]);
        u64 available = freePages.Size();
        for (usize v = 0; v < scratchVictims.Size() && available < needed; ++v)
        {
            const u32 victimIndex = scratchVictims[v].texture;
            const Texture& other = *textures[victimIndex];
            if (victimIndex == requester || !other.live || other.inFlight != 0)
            {
                continue;
            }
            const bool takeNeeded = requesterImportance > Importance(other) * 1.25f + 0.5f;
            u32 level = other.residentLevel;
            u32 take = 0;
            while (level < other.tailLevel && available < needed)
            {
                if (level >= other.wantedLevel && !takeNeeded)
                {
                    break;
                }
                available += other.pages[level].Size();
                ++level;
                ++take;
            }
            if (take > 0 && !scratchPlan.TryPushBack(EvictionStep{victimIndex, take}))
            {
                return false;
            }
        }
        return available >= needed;
    }

    // Victims in eviction order: textures holding surplus levels first, then
    // the least important (unused textures rank below every used one).
    void BuildVictims() noexcept
    {
        scratchVictims.Clear();
        for (usize i = 0; i < textures.Size(); ++i)
        {
            Texture* texture = textures[i];
            if (texture == nullptr || !texture->live ||
                texture->residentLevel >= texture->tailLevel || texture->inFlight != 0)
            {
                continue;
            }
            (void)scratchVictims.TryPushBack(Candidate{static_cast<u32>(i), Importance(*texture)});
        }
        std::sort(scratchVictims.Data(), scratchVictims.Data() + scratchVictims.Size(),
                  [this](const Candidate& lhs, const Candidate& rhs) noexcept
                  {
                      const Texture& a = *textures[lhs.texture];
                      const Texture& b = *textures[rhs.texture];
                      const bool surplusA = a.residentLevel < a.wantedLevel;
                      const bool surplusB = b.residentLevel < b.wantedLevel;
                      if (surplusA != surplusB)
                      {
                          return surplusA;
                      }
                      if (lhs.priority != rhs.priority)
                      {
                          return lhs.priority < rhs.priority;
                      }
                      return lhs.texture < rhs.texture;
                  });
    }

    void RefreshWanted() noexcept
    {
        for (usize i = 0; i < textures.Size(); ++i)
        {
            Texture* texture = textures[i];
            if (texture == nullptr || !texture->live)
            {
                continue;
            }
            if (!IsUsed(*texture))
            {
                texture->wantedLevel = texture->tailLevel;
                texture->coverage = 0.0f;
                continue;
            }
            if (texture->reportedThisFrame)
            {
                const f32 biased = texture->reportedMip + desc.mipBias;
                const f32 clamped =
                    std::clamp(std::floor(biased), 0.0f, static_cast<f32>(texture->tailLevel));
                texture->wantedLevel = static_cast<u32>(clamped);
            }
            texture->reportedThisFrame = false;
        }
    }

    void Gather(StreamingStats& out) const noexcept
    {
        out.budgetBytes = desc.budgetBytes;
        out.pageBytes = desc.pageBytes;
        out.totalPages = totalPages;
        out.usedPages = totalPages - static_cast<u32>(freePages.Size());
        out.stagingBytes = stagingBytes;
        out.residentBytes = 0;
        out.tailBytes = 0;
        out.fullChainBytes = 0;
        out.textures = 0;
        out.missingLevels = 0;
        out.texturesWithoutTail = 0;
        out.reservedPages = 0;
        out.inFlight = 0;
        for (usize i = 0; i < requests.Size(); ++i)
        {
            if (requests[i].status != RequestStatus::Free)
            {
                ++out.inFlight;
                out.reservedPages += requests[i].pages;
            }
        }
        for (usize i = 0; i < textures.Size(); ++i)
        {
            const Texture* texture = textures[i];
            if (texture == nullptr || !texture->live)
            {
                continue;
            }
            ++out.textures;
            out.fullChainBytes += texture->BytesFrom(0, texture->levelCount);
            for (u32 level = 0; level < texture->levelCount; ++level)
            {
                if (texture->state[level] == LevelState::Resident)
                {
                    out.residentBytes += texture->levels[level].bytes;
                    if (level >= texture->tailLevel)
                    {
                        out.tailBytes += texture->levels[level].bytes;
                    }
                }
            }
            if (!texture->TailResident())
            {
                ++out.texturesWithoutTail;
            }
            else if (IsUsed(*texture) && texture->residentLevel > texture->wantedLevel)
            {
                out.missingLevels += texture->residentLevel - texture->wantedLevel;
            }
        }
    }

    void Drain(TextureUploadSink& sink) noexcept
    {
        mutex.Lock();
        for (usize i = 0; i < completed.Size(); ++i)
        {
            (void)ready.TryPushBack(completed[i]);  // capacity reserved at init
        }
        completed.Clear();
        mutex.Unlock();

        // Upload in priority order within the per-frame byte budget.
        std::sort(ready.Data(), ready.Data() + ready.Size(),
                  [this](u32 lhs, u32 rhs) noexcept
                  {
                      return requests[lhs].priority > requests[rhs].priority ||
                             (requests[lhs].priority == requests[rhs].priority && lhs < rhs);
                  });
        u64 uploaded = 0;
        usize applied = 0;
        while (applied < ready.Size())
        {
            const Request& request = requests[ready[applied]];
            if (applied > 0 && uploaded + request.bytes > desc.maxUploadBytesPerFrame)
            {
                break;
            }
            uploaded += request.bytes;
            Apply(ready[applied], sink);
            ++applied;
        }
        // Keep the remainder for the next frame, preserving order.
        usize write = 0;
        for (usize i = applied; i < ready.Size(); ++i)
        {
            ready[write++] = ready[i];
        }
        while (ready.Size() > write)
        {
            ready.PopBack();
        }
    }
};

// ---------------------------------------------------------------------------

TextureStreamer::~TextureStreamer()
{
    Shutdown();
}

core::Status<core::Error> TextureStreamer::Initialize(const TextureStreamerDesc& desc,
                                                      memory::Allocator& allocator) noexcept
{
    if (m_impl != nullptr)
    {
        return core::Status<core::Error>::Err(
            MakeError(core::ErrorCode::AlreadyExists, "texture streamer is already initialized"));
    }
    if (desc.pageBytes == 0 || desc.budgetBytes < desc.pageBytes || desc.maxInFlight == 0 ||
        desc.ioThreads == 0 || desc.ioThreads > 16)
    {
        return core::Status<core::Error>::Err(
            MakeError(core::ErrorCode::InvalidArgument, "invalid texture streamer description"));
    }
    void* memory = allocator.Allocate(sizeof(Impl), alignof(Impl));
    if (memory == nullptr)
    {
        return core::Status<core::Error>::Err(
            MakeError(core::ErrorCode::OutOfRange, "out of memory for the texture streamer"));
    }
    Impl* impl = new (memory) Impl(allocator);
    impl->desc = desc;
    impl->totalPages =
        static_cast<u32>(std::min<u64>(desc.budgetBytes / desc.pageBytes, 0x7FFFFFFFu));
    impl->stats.budgetBytes = desc.budgetBytes;
    bool ok = impl->freePages.TryReserve(impl->totalPages) &&
              impl->requests.TryResize(desc.maxInFlight) &&
              impl->queue.TryReserve(desc.maxInFlight) &&
              impl->completed.TryReserve(desc.maxInFlight) &&
              impl->ready.TryReserve(desc.maxInFlight) && impl->threads.TryReserve(desc.ioThreads);
    for (u32 page = impl->totalPages; ok && page > 0; --page)
    {
        ok = impl->freePages.TryPushBack(page - 1);
    }
    m_impl = impl;
    for (u32 index = 0; ok && index < desc.ioThreads; ++index)
    {
        void* threadMemory =
            allocator.Allocate(sizeof(platform::Thread), alignof(platform::Thread));
        if (threadMemory == nullptr)
        {
            ok = false;
            break;
        }
        auto* thread = new (threadMemory) platform::Thread();
        (void)impl->threads.TryPushBack(thread);
        ok = thread->Start(allocator, "texture_io", &Impl::IoThreadEntry, impl).IsOk();
    }
    if (!ok)
    {
        Shutdown();
        return core::Status<core::Error>::Err(
            MakeError(core::ErrorCode::Internal, "texture streamer initialization failed"));
    }
    return core::Status<core::Error>::Ok();
}

void TextureStreamer::Shutdown() noexcept
{
    if (m_impl == nullptr)
    {
        return;
    }
    Impl* impl = m_impl;
    impl->mutex.Lock();
    impl->stopping = true;
    impl->workAvailable.NotifyAll();
    impl->mutex.Unlock();
    for (usize i = 0; i < impl->threads.Size(); ++i)
    {
        platform::Thread* thread = impl->threads[i];
        if (thread->Joinable())
        {
            thread->Join();
        }
        thread->~Thread();
        impl->allocator->Deallocate(thread);
    }
    for (usize i = 0; i < impl->requests.Size(); ++i)
    {
        if (impl->requests[i].staging != nullptr)
        {
            impl->allocator->Deallocate(impl->requests[i].staging);
        }
    }
    for (usize i = 0; i < impl->stagingPool.Size(); ++i)
    {
        {
            impl->allocator->Deallocate(impl->stagingPool[i].data);
        }
    }
    for (usize i = 0; i < impl->textures.Size(); ++i)
    {
        if (impl->textures[i] != nullptr)
        {
            impl->textures[i]->~Texture();
            impl->allocator->Deallocate(impl->textures[i]);
        }
    }
    memory::Allocator* allocator = impl->allocator;
    impl->~Impl();
    allocator->Deallocate(impl);
    m_impl = nullptr;
}

core::Result<TextureHandle, core::Error> TextureStreamer::Register(TextureMipSource& source,
                                                                   const char* name) noexcept
{
    using ResultType = core::Result<TextureHandle, core::Error>;
    if (m_impl == nullptr)
    {
        return ResultType::Err(
            MakeError(core::ErrorCode::InvalidArgument, "streamer is not initialized"));
    }
    const u32 levelCount = source.LevelCount();
    if (levelCount == 0 || levelCount > kMaxStreamedLevels)
    {
        return ResultType::Err(
            MakeError(core::ErrorCode::InvalidArgument, "invalid texture level count"));
    }
    Impl& impl = *m_impl;

    // Reuse a dead slot whose reads have all drained.
    u32 index = static_cast<u32>(impl.textures.Size());
    for (usize i = 0; i < impl.textures.Size(); ++i)
    {
        if (impl.textures[i] != nullptr && !impl.textures[i]->live &&
            impl.textures[i]->inFlight == 0)
        {
            index = static_cast<u32>(i);
            break;
        }
    }
    Impl::Texture* texture = nullptr;
    u32 generation = 1;
    if (index == impl.textures.Size())
    {
        void* memory = impl.allocator->Allocate(sizeof(Impl::Texture), alignof(Impl::Texture));
        if (memory == nullptr || !impl.textures.TryPushBack(nullptr))
        {
            impl.allocator->Deallocate(memory);
            return ResultType::Err(
                MakeError(core::ErrorCode::OutOfRange, "out of memory for a texture"));
        }
        texture = new (memory) Impl::Texture(*impl.allocator);
        impl.textures[index] = texture;
    }
    else
    {
        texture = impl.textures[index];
        generation = texture->generation + 1;
        texture->~Texture();
        new (texture) Impl::Texture(*impl.allocator);
    }

    texture->source = &source;
    texture->generation = generation;
    texture->levelCount = levelCount;
    const char* label = name != nullptr ? name : "";
    const usize labelLength = std::min(std::strlen(label), sizeof(texture->name) - 1);
    std::memcpy(texture->name, label, labelLength);
    texture->name[labelLength] = '\0';
    for (u32 level = 0; level < levelCount; ++level)
    {
        texture->levels[level] = source.Level(level);
    }
    // The packed tail: the coarsest run of levels each smaller than a page.
    // A texture whose every level is that small is one tail.
    u32 tail = levelCount - 1;
    while (tail > 0 && texture->levels[tail - 1].bytes < impl.desc.pageBytes)
    {
        --tail;
    }
    // A GPU-backed source knows where its driver packs mips; that boundary
    // wins, because the packed mips can only be mapped together.
    const u32 sourceTail = source.TailLevel();
    if (sourceTail < levelCount)
    {
        tail = sourceTail;
    }
    texture->tailLevel = tail;
    texture->residentLevel = levelCount;
    texture->wantedLevel = tail;
    texture->live = true;
    return ResultType::Ok(TextureHandle{index, generation});
}

void TextureStreamer::Unregister(TextureHandle handle, TextureUploadSink& sink) noexcept
{
    if (m_impl == nullptr)
    {
        return;
    }
    Impl& impl = *m_impl;
    Impl::Texture* texture = impl.Resolve(handle);
    if (texture == nullptr)
    {
        return;
    }
    texture->live = false;
    // The source must not be touched after this returns: wait for its read.
    while (texture->inFlight != 0)
    {
        impl.mutex.Lock();
        while (impl.completed.Empty() && impl.ready.Empty())
        {
            impl.workFinished.Wait(impl.mutex);
        }
        impl.mutex.Unlock();
        impl.Drain(sink);
    }
    for (u32 level = 0; level < texture->levelCount; ++level)
    {
        if (texture->state[level] == LevelState::Resident && !texture->pages[level].Empty())
        {
            const memory::DynamicArray<u32>& pages = texture->pages[level];
            sink.EvictLevel(handle, level, memory::Span<const u32>(pages.Data(), pages.Size()));
        }
        impl.FreePages(texture->pages[level]);
        texture->state[level] = LevelState::Absent;
    }
    texture->source = nullptr;
}

void TextureStreamer::ReportUsage(TextureHandle handle, f32 wantedMip, f32 screenCoverage) noexcept
{
    if (m_impl == nullptr)
    {
        return;
    }
    Impl::Texture* texture = m_impl->Resolve(handle);
    if (texture == nullptr)
    {
        return;
    }
    const f32 mip =
        std::isfinite(wantedMip) ? std::max(wantedMip, 0.0f) : static_cast<f32>(kMaxStreamedLevels);
    const f32 coverage =
        std::isfinite(screenCoverage) ? std::clamp(screenCoverage, 0.0f, 1.0f) : 0.0f;
    const u64 frame = m_impl->frame + 1;  // the frame the next Update completes
    if (!texture->reportedThisFrame)
    {
        texture->reportedMip = mip;
        texture->coverage = coverage;
        texture->reportedThisFrame = true;
    }
    else
    {
        texture->reportedMip = std::min(texture->reportedMip, mip);
        texture->coverage = std::min(texture->coverage + coverage, 1.0f);
    }
    texture->lastUsedFrame = frame;
}

StreamingStats TextureStreamer::Update(TextureUploadSink& sink) noexcept
{
    if (m_impl == nullptr)
    {
        return StreamingStats{};
    }
    Impl& impl = *m_impl;
    ++impl.frame;
    StreamingStats& stats = impl.stats;
    stats.uploadsThisFrame = 0;
    stats.uploadBytesThisFrame = 0;
    stats.evictionsThisFrame = 0;
    stats.requestsThisFrame = 0;
    stats.ioErrorsThisFrame = 0;
    stats.budgetBlockedThisFrame = 0;

    impl.Drain(sink);
    impl.RefreshWanted();
    impl.Schedule(sink);
    impl.Gather(stats);
    return stats;
}

u32 TextureStreamer::ResidentLevel(TextureHandle handle) const noexcept
{
    const Impl::Texture* texture = m_impl != nullptr ? m_impl->Resolve(handle) : nullptr;
    return texture != nullptr ? texture->residentLevel : kNoLevel;
}

bool TextureStreamer::IsValid(TextureHandle handle) const noexcept
{
    return m_impl != nullptr && m_impl->Resolve(handle) != nullptr;
}

const StreamingStats& TextureStreamer::Stats() const noexcept
{
    static const StreamingStats empty{};
    return m_impl != nullptr ? m_impl->stats : empty;
}

bool TextureStreamer::Snapshot(memory::DynamicArray<TextureResidency>& out) const noexcept
{
    out.Clear();
    if (m_impl == nullptr)
    {
        return true;
    }
    const Impl& impl = *m_impl;
    for (usize i = 0; i < impl.textures.Size(); ++i)
    {
        const Impl::Texture* texture = impl.textures[i];
        if (texture == nullptr || !texture->live)
        {
            continue;
        }
        TextureResidency entry;
        entry.handle = TextureHandle{static_cast<u32>(i), texture->generation};
        entry.name = texture->name;
        entry.levelCount = texture->levelCount;
        entry.tailLevel = texture->tailLevel;
        entry.residentLevel = texture->residentLevel;
        entry.wantedLevel = texture->wantedLevel;
        entry.loadingLevel = texture->loadingLevel;
        entry.fullChainBytes = texture->BytesFrom(0, texture->levelCount);
        for (u32 level = 0; level < texture->levelCount; ++level)
        {
            if (texture->state[level] == LevelState::Resident)
            {
                entry.residentBytes += texture->levels[level].bytes;
            }
        }
        entry.priority = impl.Priority(*texture);
        entry.used = impl.IsUsed(*texture);
        if (!out.TryPushBack(entry))
        {
            return false;
        }
    }
    return true;
}

void TextureStreamer::Flush(TextureUploadSink& sink) noexcept
{
    if (m_impl == nullptr)
    {
        return;
    }
    Impl& impl = *m_impl;
    const u64 savedBudget = impl.desc.maxUploadBytesPerFrame;
    impl.desc.maxUploadBytesPerFrame = ~0ull;
    while (true)
    {
        (void)Update(sink);
        if (impl.stats.inFlight == 0)
        {
            break;
        }
        impl.mutex.Lock();
        while (impl.completed.Empty() && impl.ready.Empty())
        {
            impl.workFinished.Wait(impl.mutex);
        }
        impl.mutex.Unlock();
    }
    impl.desc.maxUploadBytesPerFrame = savedBudget;
}

// ---------------------------------------------------------------------------
// Residency view

namespace
{

void FillRect(memory::Span<Byte> rgba, u32 width, u32 height, u32 x0, u32 y0, u32 w, u32 h,
              u32 color) noexcept
{
    const u32 x1 = std::min(x0 + w, width);
    const u32 y1 = std::min(y0 + h, height);
    for (u32 y = y0; y < y1; ++y)
    {
        Byte* row = rgba.Data() + static_cast<usize>(y) * width * 4u;
        for (u32 x = x0; x < x1; ++x)
        {
            std::memcpy(row + x * 4u, &color, 4);
        }
    }
}

}  // namespace

bool RenderResidencyView(memory::Span<const TextureResidency> textures, const StreamingStats& stats,
                         u32 width, u32 height, memory::Span<Byte> rgba,
                         const ResidencyViewColors& colors) noexcept
{
    constexpr u32 kBarHeight = 12;
    constexpr u32 kMargin = 4;
    if (width < 64 || height < kBarHeight + 16 ||
        rgba.Size() < static_cast<usize>(width) * height * 4u)
    {
        return false;
    }
    FillRect(rgba, width, height, 0, 0, width, height, colors.background);

    // Budget bar: resident pages, then pages reserved for in-flight reads.
    const u32 barWidth = width - 2 * kMargin;
    const f32 total = stats.totalPages > 0 ? static_cast<f32>(stats.totalPages) : 1.0f;
    const u32 residentPages =
        stats.usedPages > stats.reservedPages ? stats.usedPages - stats.reservedPages : 0;
    const u32 usedWidth = static_cast<u32>(barWidth * std::min(residentPages / total, 1.0f));
    const u32 reservedWidth =
        static_cast<u32>(barWidth * std::min(stats.reservedPages / total, 1.0f));
    FillRect(rgba, width, height, kMargin, kMargin, barWidth, kBarHeight - 4, colors.budgetFree);
    FillRect(rgba, width, height, kMargin, kMargin, usedWidth, kBarHeight - 4, colors.budgetUsed);
    FillRect(rgba, width, height, kMargin + usedWidth, kMargin, reservedWidth, kBarHeight - 4,
             colors.budgetReserved);

    if (textures.Empty())
    {
        return true;
    }
    // Cells laid out in a near-square grid that fits the remaining area.
    const u32 top = kBarHeight + kMargin;
    const u32 areaHeight = height - top;
    const u32 count = static_cast<u32>(textures.Size());
    u32 columns = 1;
    while (columns * columns * areaHeight < count * width && columns < count)
    {
        ++columns;
    }
    const u32 rows = (count + columns - 1) / columns;
    const u32 cellWidth = width / columns;
    const u32 cellHeight = std::max(areaHeight / std::max(rows, 1u), 3u);
    for (u32 index = 0; index < count; ++index)
    {
        const TextureResidency& texture = textures[index];
        const u32 cellX = (index % columns) * cellWidth;
        const u32 cellY = top + (index / columns) * cellHeight;
        const u32 pad = cellWidth > 8 && cellHeight > 6 ? 1u : 0u;
        const u32 stripHeight = cellHeight > 4 ? cellHeight - 2 * pad - 2 : cellHeight;
        // One box per streamed level plus one for the whole packed tail: the
        // tail is resident or not as a unit, and drawing its many tiny levels
        // separately would crowd out the levels that actually stream.
        const u32 boxes = std::min(texture.tailLevel, texture.levelCount) + 1u;
        const u32 boxWidth = std::max((cellWidth - 2 * pad) / boxes, 1u);
        for (u32 level = 0; level < boxes && level < texture.levelCount; ++level)
        {
            u32 color = colors.absent;
            const bool resident =
                level >= texture.residentLevel && texture.residentLevel < texture.levelCount;
            if (level >= texture.tailLevel)
            {
                color = resident ? colors.tail
                                 : (texture.loadingLevel == texture.tailLevel ? colors.loading
                                                                              : colors.missing);
            }
            else if (level == texture.loadingLevel)
            {
                color = colors.loading;
            }
            else if (resident)
            {
                color =
                    level < texture.wantedLevel ? colors.residentSurplus : colors.residentWanted;
            }
            else if (texture.used && level >= texture.wantedLevel)
            {
                color = colors.missing;
            }
            const u32 gap = boxWidth > 2 ? 1u : 0u;
            FillRect(rgba, width, height, cellX + pad + level * boxWidth, cellY + pad,
                     boxWidth - gap, stripHeight, color);
        }
        if (cellHeight > 4 && texture.fullChainBytes > 0)
        {
            const f32 share =
                static_cast<f32>(texture.residentBytes) / static_cast<f32>(texture.fullChainBytes);
            FillRect(rgba, width, height, cellX + pad, cellY + pad + stripHeight,
                     static_cast<u32>((cellWidth - 2 * pad) * std::min(share, 1.0f)), 1,
                     colors.budgetUsed);
        }
    }
    return true;
}

}  // namespace cyber::renderer
