#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/span.h>

namespace cyber::renderer
{

using cyber::core::Error;
using cyber::core::f32;
using cyber::core::Result;
using cyber::core::Status;
using cyber::core::u32;

inline constexpr u32 kInvalidClusterIndex = 0xFFFFFFFFU;

/// View-space bound for a local light. View-space +Z is forward and range is
/// a conservative sphere radius; spot lights use their range sphere as well.
struct ClusterLightBound
{
    f32 x = 0.0f;
    f32 y = 0.0f;
    f32 depth = 0.0f;
    f32 range = 0.0f;
};

/// Perspective cluster grid. Depth is divided logarithmically between the
/// camera near and far planes.
struct ClusterGridDesc
{
    u32 widthPixels = 0;
    u32 heightPixels = 0;
    u32 tileSizePixels = 16;
    u32 depthSlices = 24;
    f32 verticalFovRadians = 1.04719755f;
    f32 nearPlane = 0.1f;
    f32 farPlane = 1000.0f;
};

struct ClusterProjectedBounds
{
    u32 minTileX = 0;
    u32 maxTileX = 0;
    u32 minTileY = 0;
    u32 maxTileY = 0;
    u32 minSlice = 0;
    u32 maxSlice = 0;
    bool visible = false;
};

/// Compact CSR light lists. Light indices refer to the input span supplied to
/// BuildClusterLightLists and retain input order within every cluster.
class ClusterLightLists
{
public:
    explicit ClusterLightLists(cyber::memory::Allocator& allocator) noexcept;
    ClusterLightLists(const ClusterLightLists&) = delete;
    ClusterLightLists& operator=(const ClusterLightLists&) = delete;
    ClusterLightLists(ClusterLightLists&&) noexcept = default;
    ClusterLightLists& operator=(ClusterLightLists&&) noexcept = default;

    [[nodiscard]] const ClusterGridDesc& Grid() const noexcept;
    [[nodiscard]] u32 TilesX() const noexcept;
    [[nodiscard]] u32 TilesY() const noexcept;
    [[nodiscard]] u32 ClusterCount() const noexcept;
    [[nodiscard]] u32 TileCount() const noexcept;
    [[nodiscard]] cyber::memory::Span<const u32> Lights(u32 clusterIndex) const noexcept;
    /// Locate a fragment's cluster using pixel-space coordinates and positive
    /// view depth. The far plane belongs to the final slice; invalid/outside
    /// coordinates return kInvalidClusterIndex.
    [[nodiscard]] u32 ClusterAtPixel(f32 pixelX, f32 pixelY, f32 viewDepth) const noexcept;

private:
    friend Status<Error> BuildClusterLightLists(cyber::memory::Span<const ClusterLightBound>,
                                                const ClusterGridDesc&,
                                                ClusterLightLists&) noexcept;
    friend Status<Error> BuildClusterHeatMap(const ClusterLightLists&,
                                             cyber::memory::Span<u32>) noexcept;

    cyber::memory::DynamicArray<u32> m_offsets;
    cyber::memory::DynamicArray<u32> m_indices;
    cyber::memory::DynamicArray<u32> m_counts;
    cyber::memory::DynamicArray<u32> m_cursors;
    cyber::memory::DynamicArray<u32> m_scratchOffsets;
    cyber::memory::DynamicArray<u32> m_scratchIndices;
    cyber::memory::DynamicArray<ClusterProjectedBounds> m_projected;
    cyber::memory::Allocator* m_allocator = nullptr;
    ClusterGridDesc m_grid{};
    u32 m_tilesX = 0;
    u32 m_tilesY = 0;
    u32 m_clusterCount = 0;
};

/// Builds deterministic, conservative point/spot-light lists on the CPU.
/// A light is included whenever its range sphere can overlap a cluster's
/// projected tile and logarithmic depth interval. Invalid input is rejected;
/// overflow and allocation errors leave `output` unchanged.
[[nodiscard]] Status<Error>
BuildClusterLightLists(cyber::memory::Span<const ClusterLightBound> lights,
                       const ClusterGridDesc& grid, ClusterLightLists& output) noexcept;

/// Writes maximum cluster occupancy per XY tile across depth slices. A
/// renderer can map these counts to its palette without coupling this module
/// to a presentation color space. Output size must equal lists.TileCount().
[[nodiscard]] Status<Error> BuildClusterHeatMap(const ClusterLightLists& lists,
                                                cyber::memory::Span<u32> output) noexcept;

}  // namespace cyber::renderer
