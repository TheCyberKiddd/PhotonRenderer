#include <cyber/renderer/meshlet.h>

#include <cmath>
#include <limits>

namespace cyber::renderer
{

namespace
{

using cyber::core::ErrorCode;
using cyber::core::usize;
using cyber::math::Vec3;
using cyber::memory::Span;

[[nodiscard]] Status<Error> Fail(ErrorCode code, const char* message) noexcept
{
    return Status<Error>::Err(Error{code, message});
}

// How far the measured cone is widened, as a cosine. The cone's normals are
// recomputed from rounded positions wherever it is tested, so a cone fitted
// exactly would reject a triangle that rounding puts a hair outside it.
inline constexpr f32 kConeSlack = 1.0e-3f;
// Narrowest spread, as the cosine of the widest normal from the axis, that
// still yields a useful cone. Below it almost nothing can be culled and the
// test only costs time.
inline constexpr f32 kMinimumConeCosine = 0.1f;

[[nodiscard]] MeshletBounds ComputeBounds(Span<const Vec3> positions, const u32* vertices,
                                          u32 vertexCount, const u8* triangles,
                                          u32 triangleCount) noexcept
{
    MeshletBounds bounds{};
    Vec3 minimum{std::numeric_limits<f32>::max()};
    Vec3 maximum{-std::numeric_limits<f32>::max()};
    for (u32 index = 0; index < vertexCount; ++index)
    {
        const Vec3 p = positions[vertices[index]];
        minimum =
            Vec3{std::fmin(minimum.x, p.x), std::fmin(minimum.y, p.y), std::fmin(minimum.z, p.z)};
        maximum =
            Vec3{std::fmax(maximum.x, p.x), std::fmax(maximum.y, p.y), std::fmax(maximum.z, p.z)};
    }
    bounds.center = (minimum + maximum) * 0.5f;
    f32 radiusSquared = 0.0f;
    for (u32 index = 0; index < vertexCount; ++index)
    {
        radiusSquared = std::fmax(
            radiusSquared, cyber::math::LengthSquared(positions[vertices[index]] - bounds.center));
    }
    // One part in a million covers the rounding of the square root and of the
    // center itself.
    bounds.radius = std::sqrt(radiusSquared) * (1.0f + 1.0e-6f) + 1.0e-7f;

    Vec3 axis{};
    u32 normals = 0;
    for (u32 triangle = 0; triangle < triangleCount; ++triangle)
    {
        const Vec3 a = positions[vertices[triangles[triangle * 3 + 0]]];
        const Vec3 b = positions[vertices[triangles[triangle * 3 + 1]]];
        const Vec3 c = positions[vertices[triangles[triangle * 3 + 2]]];
        const Vec3 normal = cyber::math::Cross(b - a, c - a);
        const f32 length = cyber::math::Length(normal);
        if (length > 0.0f)
        {
            axis = axis + normal / length;
            ++normals;
        }
    }
    const f32 axisLength = cyber::math::Length(axis);
    if (normals == 0 || axisLength <= 1.0e-6f)
    {
        return bounds;
    }
    axis = axis / axisLength;
    f32 minimumDot = 1.0f;
    for (u32 triangle = 0; triangle < triangleCount; ++triangle)
    {
        const Vec3 a = positions[vertices[triangles[triangle * 3 + 0]]];
        const Vec3 b = positions[vertices[triangles[triangle * 3 + 1]]];
        const Vec3 c = positions[vertices[triangles[triangle * 3 + 2]]];
        const Vec3 normal = cyber::math::Cross(b - a, c - a);
        const f32 length = cyber::math::Length(normal);
        if (length > 0.0f)
        {
            minimumDot = std::fmin(minimumDot, cyber::math::Dot(normal / length, axis));
        }
    }
    minimumDot -= kConeSlack;
    if (minimumDot <= kMinimumConeCosine)
    {
        return bounds;
    }
    bounds.coneAxis = axis;
    bounds.coneCutoff = std::sqrt(1.0f - minimumDot * minimumDot);
    return bounds;
}

}  // namespace

MeshletMesh::MeshletMesh(cyber::memory::Allocator& allocator) noexcept
    : allocator(&allocator), meshlets(allocator), bounds(allocator), vertices(allocator),
      triangles(allocator)
{
}

Status<Error> BuildMeshlets(Span<const Vec3> positions, Span<const u32> indices,
                            const MeshletLimits& limits, MeshletMesh& output) noexcept
{
    if (limits.maxVertices < 3 || limits.maxVertices > kMeshletMaxVertices ||
        limits.maxTriangles < 1 || limits.maxTriangles > kMeshletMaxTriangles)
    {
        return Fail(ErrorCode::InvalidArgument, "meshlets: limits are outside 3-64 / 1-124");
    }
    if (indices.Size() % 3 != 0)
    {
        return Fail(ErrorCode::InvalidArgument, "meshlets: the index count is not a multiple of 3");
    }
    if (positions.Size() > std::numeric_limits<u32>::max())
    {
        return Fail(ErrorCode::InvalidArgument, "meshlets: too many vertices");
    }
    for (usize index = 0; index < indices.Size(); ++index)
    {
        if (indices[index] >= positions.Size())
        {
            return Fail(ErrorCode::InvalidArgument, "meshlets: an index is outside the vertices");
        }
    }

    // Built into scratch arrays and moved into `output` only on success.
    cyber::memory::Allocator& allocator = *output.allocator;
    MeshletMesh built(allocator);
    // `stamp[v] == meshlet + 1` marks vertex v as already in the current
    // meshlet at local index `local[v]`; the stamp makes the reset free.
    cyber::memory::DynamicArray<u32> stamp(allocator);
    cyber::memory::DynamicArray<u8> local(allocator);
    if (!stamp.TryResize(positions.Size()) || !local.TryResize(positions.Size()) ||
        !built.triangles.TryReserve(indices.Size()) || !built.vertices.TryReserve(indices.Size()))
    {
        return Fail(ErrorCode::Internal, "meshlets: allocation failed");
    }
    for (usize index = 0; index < stamp.Size(); ++index)
    {
        stamp[index] = 0;
    }

    Meshlet current{};
    const auto close = [&]() noexcept -> bool
    {
        if (current.triangleCount == 0)
        {
            return true;
        }
        const MeshletBounds bounds = ComputeBounds(
            positions, built.vertices.Data() + current.vertexOffset, current.vertexCount,
            built.triangles.Data() + static_cast<usize>(current.triangleOffset) * 3,
            current.triangleCount);
        if (!built.meshlets.TryPushBack(current) || !built.bounds.TryPushBack(bounds))
        {
            return false;
        }
        current.vertexOffset = static_cast<u32>(built.vertices.Size());
        current.vertexCount = 0;
        current.triangleOffset = static_cast<u32>(built.triangles.Size() / 3);
        current.triangleCount = 0;
        return true;
    };

    const usize triangleCount = indices.Size() / 3;
    for (usize triangle = 0; triangle < triangleCount; ++triangle)
    {
        const u32 corners[3] = {indices[triangle * 3 + 0], indices[triangle * 3 + 1],
                                indices[triangle * 3 + 2]};
        const u32 meshletStamp = static_cast<u32>(built.meshlets.Size()) + 1;
        u32 added = 0;
        for (u32 corner = 0; corner < 3; ++corner)
        {
            const u32 vertex = corners[corner];
            bool repeated = stamp[vertex] == meshletStamp;
            // A corner repeated inside the triangle itself counts once.
            for (u32 earlier = 0; earlier < corner && !repeated; ++earlier)
            {
                repeated = corners[earlier] == vertex;
            }
            added += repeated ? 0U : 1U;
        }
        if (current.vertexCount + added > limits.maxVertices ||
            current.triangleCount + 1 > limits.maxTriangles)
        {
            if (!close())
            {
                return Fail(ErrorCode::Internal, "meshlets: allocation failed");
            }
        }
        const u32 activeStamp = static_cast<u32>(built.meshlets.Size()) + 1;
        for (u32 corner = 0; corner < 3; ++corner)
        {
            const u32 vertex = corners[corner];
            if (stamp[vertex] != activeStamp)
            {
                stamp[vertex] = activeStamp;
                local[vertex] = static_cast<u8>(current.vertexCount);
                ++current.vertexCount;
                (void)built.vertices.TryPushBack(vertex);
            }
            (void)built.triangles.TryPushBack(local[vertex]);
        }
        ++current.triangleCount;
    }
    if (!close())
    {
        return Fail(ErrorCode::Internal, "meshlets: allocation failed");
    }
    output = static_cast<MeshletMesh&&>(built);
    return Status<Error>::Ok();
}

MeshletVisibility CullMeshlet(const MeshletBounds& bounds, const MeshletCullView& view) noexcept
{
    if (view.frustumCulling)
    {
        for (const cyber::math::Plane& plane : view.frustum.planes)
        {
            if (cyber::math::DistanceTo(plane, bounds.center) < -bounds.radius)
            {
                return MeshletVisibility::FrustumCulled;
            }
        }
    }
    if (view.coneCulling && bounds.coneCutoff < 1.0f)
    {
        const Vec3 offset = bounds.center - view.cameraPosition;
        const f32 distance = cyber::math::Length(offset);
        if (cyber::math::Dot(offset, bounds.coneAxis) >=
            bounds.coneCutoff * distance + bounds.radius)
        {
            return MeshletVisibility::BackfaceCulled;
        }
    }
    return MeshletVisibility::Visible;
}

Status<Error> CullMeshletsReference(const MeshletMesh& mesh, const MeshletCullView& view,
                                    cyber::memory::DynamicArray<u32>& survivors,
                                    MeshletCullStats* stats) noexcept
{
    survivors.Clear();
    if (!survivors.TryReserve(mesh.meshlets.Size()))
    {
        return Fail(ErrorCode::Internal, "meshlets: allocation failed");
    }
    MeshletCullStats counts{};
    for (usize index = 0; index < mesh.meshlets.Size(); ++index)
    {
        ++counts.tested;
        switch (CullMeshlet(mesh.bounds[index], view))
        {
        case MeshletVisibility::FrustumCulled:
            ++counts.frustumCulled;
            break;
        case MeshletVisibility::BackfaceCulled:
            ++counts.backfaceCulled;
            break;
        case MeshletVisibility::Visible:
            ++counts.visible;
            counts.visibleTriangles += mesh.meshlets[index].triangleCount;
            (void)survivors.TryPushBack(static_cast<u32>(index));
            break;
        }
    }
    if (stats != nullptr)
    {
        *stats = counts;
    }
    return Status<Error>::Ok();
}

Status<Error> ExpandMeshletsToIndices(const MeshletMesh& mesh, Span<const u32> meshletIndices,
                                      cyber::memory::DynamicArray<u32>& indices) noexcept
{
    usize total = 0;
    for (usize index = 0; index < meshletIndices.Size(); ++index)
    {
        if (meshletIndices[index] >= mesh.meshlets.Size())
        {
            return Fail(ErrorCode::InvalidArgument, "meshlets: a meshlet index is out of range");
        }
        total += static_cast<usize>(mesh.meshlets[meshletIndices[index]].triangleCount) * 3;
    }
    indices.Clear();
    if (!indices.TryReserve(total))
    {
        return Fail(ErrorCode::Internal, "meshlets: allocation failed");
    }
    for (usize index = 0; index < meshletIndices.Size(); ++index)
    {
        const Meshlet& meshlet = mesh.meshlets[meshletIndices[index]];
        const u8* triangles =
            mesh.triangles.Data() + static_cast<usize>(meshlet.triangleOffset) * 3;
        const u32* vertices = mesh.vertices.Data() + meshlet.vertexOffset;
        for (u32 corner = 0; corner < meshlet.triangleCount * 3; ++corner)
        {
            (void)indices.TryPushBack(vertices[triangles[corner]]);
        }
    }
    return Status<Error>::Ok();
}

}  // namespace cyber::renderer
