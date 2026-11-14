#include "scratch_allocator.h"

#include <cstdint>
#include <cstdlib>

namespace cyber::software::detail
{

namespace
{

usize AlignUp(usize value, usize alignment) noexcept
{
    return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace

ScratchAllocator::~ScratchAllocator()
{
    ReleaseChunks();
}

void ScratchAllocator::ReleaseChunks() noexcept
{
    for (usize index = 0; index < m_chunkCount; ++index)
    {
        std::free(m_chunks[index].data);
        m_chunks[index] = Chunk{};
    }
    m_chunkCount = 0;
    m_current = 0;
    m_offset = 0;
    m_used = 0;
}

bool ScratchAllocator::AddChunk(usize minimumBytes) noexcept
{
    if (m_chunkCount == kMaxChunks)
    {
        return false;
    }
    usize size = kMinChunkBytes;
    if (m_chunkCount > 0)
    {
        // Geometric growth keeps the chunk count logarithmic in the peak.
        const usize previous = m_chunks[m_chunkCount - 1].size;
        size = previous > static_cast<usize>(-1) / 2 ? previous : previous * 2;
    }
    if (size < minimumBytes)
    {
        size = minimumBytes;
    }
    auto* data = static_cast<unsigned char*>(std::malloc(size));
    if (data == nullptr)
    {
        return false;
    }
    m_chunks[m_chunkCount] = Chunk{data, size};
    ++m_chunkCount;
    return true;
}

void* ScratchAllocator::Allocate(usize size, usize alignment) noexcept
{
    if (size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0)
    {
        return nullptr;
    }

    while (true)
    {
        if (m_current < m_chunkCount)
        {
            Chunk& chunk = m_chunks[m_current];
            const auto base = reinterpret_cast<std::uintptr_t>(chunk.data);
            const usize aligned =
                AlignUp(static_cast<usize>(base) + m_offset, alignment) - static_cast<usize>(base);
            if (aligned <= chunk.size && size <= chunk.size - aligned)
            {
                m_offset = aligned + size;
                m_used += size;
                m_peak = m_used > m_peak ? m_used : m_peak;
                ++m_allocations;
                return chunk.data + aligned;
            }
            // Move to the next retained chunk, if one is large enough to be
            // worth trying; otherwise grow.
            if (m_current + 1 < m_chunkCount)
            {
                ++m_current;
                m_offset = 0;
                continue;
            }
        }

        if (size > static_cast<usize>(-1) - alignment)
        {
            return nullptr;
        }
        if (!AddChunk(size + alignment))
        {
            return nullptr;
        }
        m_current = m_chunkCount - 1;
        m_offset = 0;
    }
}

void ScratchAllocator::Deallocate(void*) noexcept {}

void ScratchAllocator::Reset() noexcept
{
    Rewind(Marker{});
}

bool ScratchAllocator::Owns(const void* pointer) const noexcept
{
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    for (usize index = 0; index < m_chunkCount; ++index)
    {
        const auto base = reinterpret_cast<std::uintptr_t>(m_chunks[index].data);
        if (address >= base && address < base + m_chunks[index].size)
        {
            return true;
        }
    }
    return false;
}

memory::AllocatorStats ScratchAllocator::Stats() const noexcept
{
    memory::AllocatorStats stats{};
    for (usize index = 0; index < m_chunkCount; ++index)
    {
        stats.capacity += m_chunks[index].size;
    }
    stats.used = m_used;
    stats.peak = m_peak;
    stats.allocations = m_allocations;
    return stats;
}

ScratchAllocator::Marker ScratchAllocator::Mark() const noexcept
{
    return Marker{m_current, m_offset};
}

void ScratchAllocator::Rewind(const Marker& marker) noexcept
{
    if (marker.chunk == 0 && marker.offset == 0)
    {
        // Back at the bottom of the stack. If a peak spilled across chunks,
        // replace them with one chunk of the combined size so the next frame
        // of the same shape is served from a single block.
        if (m_chunkCount > 1)
        {
            usize total = 0;
            for (usize index = 0; index < m_chunkCount; ++index)
            {
                total += m_chunks[index].size;
            }
            ReleaseChunks();
            (void)AddChunk(total);
        }
        m_current = 0;
        m_offset = 0;
        m_used = 0;
        return;
    }
    m_current = marker.chunk;
    m_offset = marker.offset;
    // `used` is approximate across chunk boundaries; it is a statistic only.
    usize used = marker.offset;
    for (usize index = 0; index < marker.chunk && index < m_chunkCount; ++index)
    {
        used += m_chunks[index].size;
    }
    m_used = used < m_used ? used : m_used;
}

ScratchAllocator& ScratchAllocator::ForCurrentThread() noexcept
{
    thread_local ScratchAllocator allocator;
    return allocator;
}

}  // namespace cyber::software::detail
