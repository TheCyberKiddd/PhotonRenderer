#include <cyber/renderer/clustered_lighting.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace cyber::renderer
{
namespace
{

using cyber::core::ErrorCode;
using cyber::core::u64;
using cyber::core::usize;

[[nodiscard]] Error ClusterError(ErrorCode code, const char* message) noexcept
{
    return Error{code, message};
}

[[nodiscard]] bool ValidGrid(const ClusterGridDesc& grid) noexcept
{
    return grid.widthPixels > 0 && grid.heightPixels > 0 && grid.tileSizePixels > 0 &&
           grid.depthSlices > 0 && std::isfinite(grid.verticalFovRadians) &&
           grid.verticalFovRadians > 0.0f && grid.verticalFovRadians < 3.14159265f &&
           std::isfinite(grid.nearPlane) && std::isfinite(grid.farPlane) && grid.nearPlane > 0.0f &&
           grid.farPlane > grid.nearPlane;
}

[[nodiscard]] u32 DepthSlice(f32 depth, const ClusterGridDesc& grid) noexcept
{
    if (depth >= grid.farPlane)
    {
        return grid.depthSlices - 1;
    }
    const double ratio = std::log(static_cast<double>(depth) / grid.nearPlane) /
                         std::log(static_cast<double>(grid.farPlane) / grid.nearPlane);
    const double scaled = std::floor(ratio * static_cast<double>(grid.depthSlices));
    return static_cast<u32>(std::clamp(scaled, 0.0, static_cast<double>(grid.depthSlices - 1)));
}

[[nodiscard]] ClusterProjectedBounds ProjectLight(const ClusterLightBound& light,
                                                  const ClusterGridDesc& grid, u32 tilesX,
                                                  u32 tilesY, f32 tanHalfFovY, f32 aspect) noexcept
{
    ClusterProjectedBounds result{};
    const f32 depthMin = light.depth - light.range;
    const f32 depthMax = light.depth + light.range;
    if (depthMax < grid.nearPlane || depthMin > grid.farPlane)
    {
        return result;
    }

    const f32 clippedDepthMin = std::max(depthMin, grid.nearPlane);
    const f32 clippedDepthMax = std::min(depthMax, grid.farPlane);
    result.minSlice = DepthSlice(clippedDepthMin, grid);
    result.maxSlice = DepthSlice(clippedDepthMax, grid);

    f32 pixelMinX = 0.0f;
    f32 pixelMaxX = static_cast<f32>(grid.widthPixels);
    f32 pixelMinY = 0.0f;
    f32 pixelMaxY = static_cast<f32>(grid.heightPixels);
    if (depthMin > grid.nearPlane)
    {
        const f32 depthCorners[] = {depthMin, depthMax};
        const f32 xCorners[] = {light.x - light.range, light.x + light.range};
        const f32 yCorners[] = {light.y - light.range, light.y + light.range};
        pixelMinX = std::numeric_limits<f32>::infinity();
        pixelMaxX = -std::numeric_limits<f32>::infinity();
        pixelMinY = std::numeric_limits<f32>::infinity();
        pixelMaxY = -std::numeric_limits<f32>::infinity();
        for (f32 depth : depthCorners)
        {
            for (f32 x : xCorners)
            {
                const f32 pixel = (0.5f + 0.5f * x / (depth * tanHalfFovY * aspect)) *
                                  static_cast<f32>(grid.widthPixels);
                pixelMinX = std::min(pixelMinX, pixel);
                pixelMaxX = std::max(pixelMaxX, pixel);
            }
            for (f32 y : yCorners)
            {
                const f32 pixel =
                    (0.5f - 0.5f * y / (depth * tanHalfFovY)) * static_cast<f32>(grid.heightPixels);
                pixelMinY = std::min(pixelMinY, pixel);
                pixelMaxY = std::max(pixelMaxY, pixel);
            }
        }
    }

    if (!std::isfinite(pixelMinX) || !std::isfinite(pixelMaxX) || !std::isfinite(pixelMinY) ||
        !std::isfinite(pixelMaxY))
    {
        pixelMinX = 0.0f;
        pixelMaxX = static_cast<f32>(grid.widthPixels);
        pixelMinY = 0.0f;
        pixelMaxY = static_cast<f32>(grid.heightPixels);
    }

    if (pixelMaxX < 0.0f || pixelMinX > static_cast<f32>(grid.widthPixels) || pixelMaxY < 0.0f ||
        pixelMinY > static_cast<f32>(grid.heightPixels))
    {
        return result;
    }

    const f32 tileSize = static_cast<f32>(grid.tileSizePixels);
    const f32 maxTileX = static_cast<f32>(tilesX - 1);
    const f32 maxTileY = static_cast<f32>(tilesY - 1);
    result.minTileX =
        static_cast<u32>(std::clamp(std::floor((pixelMinX - 1.0f) / tileSize), 0.0f, maxTileX));
    result.maxTileX =
        static_cast<u32>(std::clamp(std::floor((pixelMaxX + 1.0f) / tileSize), 0.0f, maxTileX));
    result.minTileY =
        static_cast<u32>(std::clamp(std::floor((pixelMinY - 1.0f) / tileSize), 0.0f, maxTileY));
    result.maxTileY =
        static_cast<u32>(std::clamp(std::floor((pixelMaxY + 1.0f) / tileSize), 0.0f, maxTileY));
    result.visible = true;
    return result;
}

template <typename Callback>
void ForEachCandidate(const ClusterProjectedBounds& bounds, u32 tilesX, u32 tilesY,
                      Callback&& callback) noexcept
{
    if (!bounds.visible)
    {
        return;
    }
    for (u32 slice = bounds.minSlice; slice <= bounds.maxSlice; ++slice)
    {
        for (u32 tileY = bounds.minTileY; tileY <= bounds.maxTileY; ++tileY)
        {
            for (u32 tileX = bounds.minTileX; tileX <= bounds.maxTileX; ++tileX)
            {
                const u32 cluster = (slice * tilesY + tileY) * tilesX + tileX;
                callback(cluster);
            }
        }
    }
}

}  // namespace

ClusterLightLists::ClusterLightLists(cyber::memory::Allocator& allocator) noexcept
    : m_offsets(allocator), m_indices(allocator), m_counts(allocator), m_cursors(allocator),
      m_scratchOffsets(allocator), m_scratchIndices(allocator), m_projected(allocator),
      m_allocator(&allocator)
{
}

const ClusterGridDesc& ClusterLightLists::Grid() const noexcept
{
    return m_grid;
}

u32 ClusterLightLists::TilesX() const noexcept
{
    return m_tilesX;
}

u32 ClusterLightLists::TilesY() const noexcept
{
    return m_tilesY;
}

u32 ClusterLightLists::ClusterCount() const noexcept
{
    return m_clusterCount;
}

u32 ClusterLightLists::TileCount() const noexcept
{
    return m_tilesX * m_tilesY;
}

cyber::memory::Span<const u32> ClusterLightLists::Lights(u32 clusterIndex) const noexcept
{
    if (clusterIndex >= m_clusterCount ||
        m_offsets.Size() != static_cast<usize>(m_clusterCount) + 1)
    {
        return {};
    }
    const u32 begin = m_offsets[clusterIndex];
    const u32 end = m_offsets[clusterIndex + 1];
    if (begin == end)
    {
        return {};
    }
    return {m_indices.Data() + begin, end - begin};
}

u32 ClusterLightLists::ClusterAtPixel(f32 pixelX, f32 pixelY, f32 viewDepth) const noexcept
{
    if (m_clusterCount == 0 || !std::isfinite(pixelX) || !std::isfinite(pixelY) ||
        !std::isfinite(viewDepth) || pixelX < 0.0f || pixelY < 0.0f ||
        pixelX >= static_cast<f32>(m_grid.widthPixels) ||
        pixelY >= static_cast<f32>(m_grid.heightPixels) || viewDepth < m_grid.nearPlane ||
        viewDepth > m_grid.farPlane)
    {
        return kInvalidClusterIndex;
    }

    const u32 tileX = static_cast<u32>(pixelX) / m_grid.tileSizePixels;
    const u32 tileY = static_cast<u32>(pixelY) / m_grid.tileSizePixels;
    const u32 slice = DepthSlice(viewDepth, m_grid);
    return (slice * m_tilesY + tileY) * m_tilesX + tileX;
}

Status<Error> BuildClusterLightLists(cyber::memory::Span<const ClusterLightBound> lights,
                                     const ClusterGridDesc& grid,
                                     ClusterLightLists& output) noexcept
{
    if (!ValidGrid(grid) || (lights.Size() != 0 && lights.Data() == nullptr))
    {
        return Status<Error>::Err(ClusterError(ErrorCode::InvalidArgument,
                                               "cluster lighting: invalid grid or light span"));
    }
    if (lights.Size() > std::numeric_limits<u32>::max())
    {
        return Status<Error>::Err(
            ClusterError(ErrorCode::OutOfRange, "cluster lighting: too many local lights"));
    }
    for (usize index = 0; index < lights.Size(); ++index)
    {
        const ClusterLightBound& light = lights[index];
        if (!std::isfinite(light.x) || !std::isfinite(light.y) || !std::isfinite(light.depth) ||
            !std::isfinite(light.range) || light.range <= 0.0f)
        {
            return Status<Error>::Err(ClusterError(ErrorCode::InvalidArgument,
                                                   "cluster lighting: invalid local-light bound"));
        }
    }

    const u64 tilesX64 =
        (static_cast<u64>(grid.widthPixels) + grid.tileSizePixels - 1) / grid.tileSizePixels;
    const u64 tilesY64 =
        (static_cast<u64>(grid.heightPixels) + grid.tileSizePixels - 1) / grid.tileSizePixels;
    const u64 tileCount64 = tilesX64 * tilesY64;
    if (tilesX64 > std::numeric_limits<u32>::max() || tilesY64 > std::numeric_limits<u32>::max() ||
        tileCount64 == 0 || tileCount64 > std::numeric_limits<u32>::max() / grid.depthSlices ||
        tileCount64 * grid.depthSlices >= std::numeric_limits<usize>::max())
    {
        return Status<Error>::Err(
            ClusterError(ErrorCode::OutOfRange, "cluster lighting: cluster grid is too large"));
    }
    const u64 clusterCount64 = tileCount64 * grid.depthSlices;
    const u32 tilesX = static_cast<u32>(tilesX64);
    const u32 tilesY = static_cast<u32>(tilesY64);
    const u32 clusterCount = static_cast<u32>(clusterCount64);
    const f32 tanHalfFovY = std::tan(grid.verticalFovRadians * 0.5f);
    const f32 aspect = static_cast<f32>(grid.widthPixels) / static_cast<f32>(grid.heightPixels);
    if (!std::isfinite(tanHalfFovY) || !(tanHalfFovY > 0.0f) || !std::isfinite(aspect) ||
        !(aspect > 0.0f))
    {
        return Status<Error>::Err(
            ClusterError(ErrorCode::InvalidArgument, "cluster lighting: invalid projection"));
    }

    if (output.m_allocator == nullptr)
    {
        return Status<Error>::Err(
            ClusterError(ErrorCode::Internal, "cluster lighting: output has no allocator"));
    }

    const usize clusterSize = static_cast<usize>(clusterCount);
    if (!output.m_counts.TryResize(clusterSize) || !output.m_cursors.TryResize(clusterSize) ||
        !output.m_offsets.TryReserve(clusterSize + 1) ||
        !output.m_scratchOffsets.TryResize(clusterSize + 1) ||
        !output.m_projected.TryResize(lights.Size()))
    {
        return Status<Error>::Err(
            ClusterError(ErrorCode::Internal, "cluster lighting: allocation failed"));
    }
    for (u32 cluster = 0; cluster < clusterCount; ++cluster)
    {
        output.m_counts[cluster] = 0;
    }

    for (usize index = 0; index < lights.Size(); ++index)
    {
        const ClusterProjectedBounds bounds =
            ProjectLight(lights[index], grid, tilesX, tilesY, tanHalfFovY, aspect);
        output.m_projected[index] = bounds;
        bool overflow = false;
        ForEachCandidate(bounds, tilesX, tilesY,
                         [&](u32 cluster) noexcept
                         {
                             if (output.m_counts[cluster] == std::numeric_limits<u32>::max())
                             {
                                 overflow = true;
                                 return;
                             }
                             ++output.m_counts[cluster];
                         });
        if (overflow)
        {
            return Status<Error>::Err(
                ClusterError(ErrorCode::Overflow, "cluster lighting: cluster count overflow"));
        }
    }

    output.m_scratchOffsets[0] = 0;
    for (u32 cluster = 0; cluster < clusterCount; ++cluster)
    {
        const u64 next =
            static_cast<u64>(output.m_scratchOffsets[cluster]) + output.m_counts[cluster];
        if (next > std::numeric_limits<u32>::max())
        {
            return Status<Error>::Err(ClusterError(ErrorCode::OutOfRange,
                                                   "cluster lighting: light-index data too large"));
        }
        output.m_scratchOffsets[cluster + 1] = static_cast<u32>(next);
    }
    const u32 totalIndices = output.m_scratchOffsets[clusterCount];
    if (!output.m_indices.TryReserve(totalIndices) ||
        !output.m_scratchIndices.TryResize(totalIndices))
    {
        return Status<Error>::Err(
            ClusterError(ErrorCode::Internal, "cluster lighting: allocation failed"));
    }
    for (u32 cluster = 0; cluster < clusterCount; ++cluster)
    {
        output.m_cursors[cluster] = output.m_scratchOffsets[cluster];
    }
    for (usize index = 0; index < lights.Size(); ++index)
    {
        ForEachCandidate(
            output.m_projected[index], tilesX, tilesY, [&](u32 cluster) noexcept
            { output.m_scratchIndices[output.m_cursors[cluster]++] = static_cast<u32>(index); });
    }

    auto oldOffsets = std::move(output.m_offsets);
    output.m_offsets = std::move(output.m_scratchOffsets);
    output.m_scratchOffsets = std::move(oldOffsets);
    auto oldIndices = std::move(output.m_indices);
    output.m_indices = std::move(output.m_scratchIndices);
    output.m_scratchIndices = std::move(oldIndices);
    output.m_grid = grid;
    output.m_tilesX = tilesX;
    output.m_tilesY = tilesY;
    output.m_clusterCount = clusterCount;
    return Status<Error>::Ok();
}

Status<Error> BuildClusterHeatMap(const ClusterLightLists& lists,
                                  cyber::memory::Span<u32> output) noexcept
{
    if (output.Size() != lists.TileCount() || (output.Size() != 0 && output.Data() == nullptr))
    {
        return Status<Error>::Err(
            ClusterError(ErrorCode::InvalidArgument, "cluster heat map: invalid output span"));
    }
    for (u32 tile = 0; tile < lists.TileCount(); ++tile)
    {
        u32 count = 0;
        for (u32 slice = 0; slice < lists.m_grid.depthSlices; ++slice)
        {
            count = std::max(
                count, static_cast<u32>(lists.Lights(slice * lists.TileCount() + tile).Size()));
        }
        output[tile] = count;
    }
    return Status<Error>::Ok();
}

}  // namespace cyber::renderer
