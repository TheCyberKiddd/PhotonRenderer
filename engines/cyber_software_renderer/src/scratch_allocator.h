#pragma once

#include <cyber/memory/allocator.h>

namespace cyber::software::detail
{

using cyber::core::usize;

// Per-thread, grow-only arena for the transient arrays a single draw builds:
// the clipped triangle list, the prepared triangles, and the tile bins.
//
// Those arrays used to be allocated from the caller's heap allocator and freed
// at the end of every draw. On Windows a freed block of that size is handed
// back to the OS, so the next draw page-faulted the same memory in again --
// about 29 faults and 0.1 ms per 576-triangle draw, paid on every draw call.
// The arena keeps its chunks for the life of the thread, so steady-state draws
// touch memory that is already committed and allocate nothing.
//
// Allocation bumps a cursor; Deallocate is a no-op. Memory is reclaimed only
// when the ScratchScope that was open at allocation time closes, so scopes
// must nest, which they do on one thread: a job that runs inside a waiting
// scope finishes before the scope resumes. Retained memory is bounded by the
// largest single draw a thread has issued.
class ScratchAllocator final : public memory::Allocator
{
public:
    struct Marker
    {
        usize chunk = 0;
        usize offset = 0;
    };

    ScratchAllocator() noexcept = default;
    ~ScratchAllocator() override;

    [[nodiscard]] void* Allocate(usize size,
                                 usize alignment = memory::kDefaultAlignment) noexcept override;
    void Deallocate(void* pointer) noexcept override;
    void Reset() noexcept override;
    [[nodiscard]] bool Owns(const void* pointer) const noexcept override;
    [[nodiscard]] memory::AllocatorStats Stats() const noexcept override;

    [[nodiscard]] Marker Mark() const noexcept;
    void Rewind(const Marker& marker) noexcept;

    // The calling thread's arena.
    [[nodiscard]] static ScratchAllocator& ForCurrentThread() noexcept;

private:
    struct Chunk
    {
        unsigned char* data = nullptr;
        usize size = 0;
    };

    static constexpr usize kMaxChunks = 32;
    static constexpr usize kMinChunkBytes = static_cast<usize>(1) << 20;

    [[nodiscard]] bool AddChunk(usize minimumBytes) noexcept;
    void ReleaseChunks() noexcept;

    Chunk m_chunks[kMaxChunks] = {};
    usize m_chunkCount = 0;
    usize m_current = 0;
    usize m_offset = 0;
    usize m_used = 0;
    usize m_peak = 0;
    usize m_allocations = 0;
};

// Rewinds the calling thread's arena to where it stood on construction.
class ScratchScope
{
public:
    ScratchScope() noexcept
        : m_allocator(ScratchAllocator::ForCurrentThread()), m_marker(m_allocator.Mark())
    {
    }
    ~ScratchScope()
    {
        m_allocator.Rewind(m_marker);
    }
    ScratchScope(const ScratchScope&) = delete;
    ScratchScope& operator=(const ScratchScope&) = delete;

    [[nodiscard]] memory::Allocator& Allocator() noexcept
    {
        return m_allocator;
    }

private:
    ScratchAllocator& m_allocator;
    ScratchAllocator::Marker m_marker;
};

}  // namespace cyber::software::detail
