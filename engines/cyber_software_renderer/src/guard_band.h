#pragma once

// Guard-band clipping, shared by the three draw paths.
//
// The rasterizer snaps positions to 24.8 fixed point and cannot represent a
// coordinate beyond +/-16384 pixels. A triangle that straddles the near plane
// projects one of its clipped corners a long way off screen, so that limit is
// reached by ordinary geometry: a floor-sized quad seen from above has a
// corner behind the camera, and the triangle that carries it lands outside the
// band. Rejecting such a triangle drops a wedge of the scene that should have
// been drawn.
//
// Clipping it instead is exact when it is done before the perspective divide:
// each guard-band edge is a plane in homogeneous clip space, and interpolating
// the clip position along it gives the same vertex the rasterizer would have
// produced had it been able to reach that far.

#include <cyber/core/types.h>
#include <cyber/math/vec4.h>
#include <cyber/software/pipeline.h>

#include <cmath>

namespace cyber::software::guard
{

using cyber::core::f32;
using cyber::core::u32;

// The rasterizer's own limit. A pixel of slack keeps a vertex that lands
// exactly on the boundary from rounding past it.
inline constexpr f32 kMaxScreenCoordinate = 16384.0f;
inline constexpr f32 kScreenLimit = kMaxScreenCoordinate - 1.0f;

// A triangle clipped by the near plane is a quad; four more planes can add one
// vertex each. The extra slots are headroom, not reachable geometry.
inline constexpr u32 kMaxGuardPolygonVertices = 12;

// The guard band as NDC bounds, which is where it becomes a set of clip-space
// planes. Screen Y runs opposite to NDC Y, so the Y bounds are swapped.
struct Band
{
    f32 minX = -1.0f;
    f32 maxX = 1.0f;
    f32 minY = -1.0f;
    f32 maxY = 1.0f;
};

[[nodiscard]] inline Band MakeBand(const Viewport& viewport) noexcept
{
    Band band;
    band.minX = 2.0f * ((-kScreenLimit - viewport.x) / viewport.width) - 1.0f;
    band.maxX = 2.0f * ((kScreenLimit - viewport.x) / viewport.width) - 1.0f;
    band.minY = 1.0f - 2.0f * ((kScreenLimit - viewport.y) / viewport.height);
    band.maxY = 1.0f - 2.0f * ((-kScreenLimit - viewport.y) / viewport.height);
    return band;
}

inline constexpr u32 kBandPlaneCount = 4;

// Signed distance to one guard-band plane; inside when the result is <= 0.
// Valid for w > 0, which the caller checks before clipping.
[[nodiscard]] inline f32 PlaneDistance(u32 plane, const math::Vec4& position,
                                       const Band& band) noexcept
{
    switch (plane)
    {
    case 0:
        return band.minX * position.w - position.x;
    case 1:
        return position.x - band.maxX * position.w;
    case 2:
        return band.minY * position.w - position.y;
    default:
        return position.y - band.maxY * position.w;
    }
}

// True when every vertex is already inside the band, which is the common case
// and skips the clip entirely.
template <typename Vertex>
[[nodiscard]] bool IsInside(const Vertex* polygon, u32 count, const Band& band) noexcept
{
    for (u32 index = 0; index < count; ++index)
    {
        for (u32 plane = 0; plane < kBandPlaneCount; ++plane)
        {
            if (PlaneDistance(plane, polygon[index].position, band) > 0.0f)
            {
                return false;
            }
        }
    }
    return true;
}

// Every w must be positive for the planes above to mean anything. After the
// near clip this holds for the projections the renderer builds, but a caller
// may hand in any matrix, so it is checked rather than assumed.
template <typename Vertex>
[[nodiscard]] bool HasPositiveW(const Vertex* polygon, u32 count) noexcept
{
    for (u32 index = 0; index < count; ++index)
    {
        const f32 w = polygon[index].position.w;
        if (!(w > 0.0f) || !math::IsFinite(w))
        {
            return false;
        }
    }
    return true;
}

// Sutherland-Hodgman against the four band planes, in place. `polygon` and
// `scratch` must both hold `kMaxGuardPolygonVertices`. `lerp(a, b, t)` returns
// the vertex at `t` along the edge, interpolating the clip position and
// whatever else the vertex carries. Returns the new vertex count, which is 0
// when nothing survived or the polygon outgrew its capacity.
template <typename Vertex, typename Lerp>
[[nodiscard]] u32 Clip(Vertex* polygon, u32 count, const Band& band, Vertex* scratch,
                       Lerp lerp) noexcept
{
    for (u32 plane = 0; plane < kBandPlaneCount && count != 0; ++plane)
    {
        u32 produced = 0;
        for (u32 index = 0; index < count; ++index)
        {
            const Vertex& current = polygon[index];
            const Vertex& next = polygon[(index + 1) % count];
            const f32 currentDistance = PlaneDistance(plane, current.position, band);
            const f32 nextDistance = PlaneDistance(plane, next.position, band);
            const bool currentInside = currentDistance <= 0.0f;
            const bool nextInside = nextDistance <= 0.0f;

            if (currentInside)
            {
                if (produced >= kMaxGuardPolygonVertices)
                {
                    return 0;
                }
                scratch[produced++] = current;
            }

            if (currentInside == nextInside)
            {
                continue;
            }

            const f32 denominator = currentDistance - nextDistance;
            if (denominator == 0.0f || !math::IsFinite(denominator))
            {
                continue;
            }
            const f32 factor = currentDistance / denominator;
            // An intersection at an endpoint is the endpoint, which this loop
            // emits already or will emit on the next edge.
            if (!(factor > 0.0f) || !(factor < 1.0f))
            {
                continue;
            }
            if (produced >= kMaxGuardPolygonVertices)
            {
                return 0;
            }
            scratch[produced++] = lerp(current, next, factor);
        }

        count = produced < 3 ? 0 : produced;
        for (u32 index = 0; index < count; ++index)
        {
            polygon[index] = scratch[index];
        }
    }
    return count;
}

}  // namespace cyber::software::guard
