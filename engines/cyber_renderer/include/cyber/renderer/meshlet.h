#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/math/frustum.h>
#include <cyber/math/vec3.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/memory/span.h>

namespace cyber::renderer
{

using cyber::core::Error;
using cyber::core::f32;
using cyber::core::Status;
using cyber::core::u32;
using cyber::core::u8;

// ---------------------------------------------------------------------------
// Meshlets (Phase 32)
//
// A meshlet is a small cluster of triangles with its own vertex list, sized
// for one mesh-shader group, plus bounds that let a whole cluster be culled
// before any of its triangles are processed: a sphere for the frustum and a
// normal cone for back faces. On hardware with mesh shaders an amplification
// shader runs `CullMeshlet` per meshlet and launches mesh-shader groups only
// for survivors. Where there are none — the adapters this phase is developed
// on — `ExpandMeshletsToIndices` turns the survivors into an ordinary index
// list for the vertex pipeline, so the same culling still removes work.
// ---------------------------------------------------------------------------

/// Vertex and triangle limits of one meshlet. 64 and 124 are the sizes the
/// D3D12 mesh-shader guidance recommends: 124 triangles keep the primitive
/// index block within 512 bytes with four bytes of headroom.
inline constexpr u32 kMeshletMaxVertices = 64;
inline constexpr u32 kMeshletMaxTriangles = 124;

/// Limits for one build. Each must be at least 3 and 1 respectively, and at
/// most the constants above.
struct MeshletLimits
{
    u32 maxVertices = kMeshletMaxVertices;
    u32 maxTriangles = kMeshletMaxTriangles;
};

/// One meshlet's ranges in the shared arrays of a `MeshletMesh`.
struct Meshlet
{
    /// First entry in `MeshletMesh::vertices`.
    u32 vertexOffset = 0;
    u32 vertexCount = 0;
    /// First triangle in `MeshletMesh::triangles`, counted in triangles.
    u32 triangleOffset = 0;
    u32 triangleCount = 0;
};

/// Culling bounds of one meshlet.
struct MeshletBounds
{
    /// Bounding sphere of the meshlet's vertices.
    cyber::math::Vec3 center{};
    f32 radius = 0.0f;
    /// Unit axis of a cone containing every triangle's normal.
    cyber::math::Vec3 coneAxis{};
    /// Sine of the cone's half-angle, widened slightly for rounding. 1 means
    /// the normals spread too far for a cone and back-face culling is off.
    f32 coneCutoff = 1.0f;
};

/// A mesh split into meshlets. Arrays use the allocator given at
/// construction; triangles index the meshlet's own vertex list with one byte
/// per corner.
struct MeshletMesh
{
    explicit MeshletMesh(cyber::memory::Allocator& allocator) noexcept;

    /// Allocator of every array below.
    cyber::memory::Allocator* allocator = nullptr;
    cyber::memory::DynamicArray<Meshlet> meshlets;
    cyber::memory::DynamicArray<MeshletBounds> bounds;
    /// Indices into the source vertex array, per meshlet.
    cyber::memory::DynamicArray<u32> vertices;
    /// Three local vertex indices per triangle.
    cyber::memory::DynamicArray<u8> triangles;
};

/// Splits an indexed triangle list into meshlets, in index order, starting a
/// new meshlet whenever the next triangle would exceed either limit, and
/// computes each meshlet's bounds.
/// @param positions Vertex positions the indices refer to.
/// @param indices Triangle list; its size must be a multiple of three and
/// every index must be inside `positions`.
/// @return Success, or InvalidArgument/Internal. A failure leaves `output`
/// unchanged.
[[nodiscard]] Status<Error> BuildMeshlets(cyber::memory::Span<const cyber::math::Vec3> positions,
                                          cyber::memory::Span<const u32> indices,
                                          const MeshletLimits& limits,
                                          MeshletMesh& output) noexcept;

/// What the amplification stage tests.
struct MeshletCullView
{
    cyber::math::Frustum frustum{};
    cyber::math::Vec3 cameraPosition{};
    bool frustumCulling = true;
    /// Back-face culling with the normal cone. Only valid for meshes drawn
    /// with back faces culled, counter-clockwise front faces.
    bool coneCulling = true;
};

/// Why a meshlet was kept or dropped.
enum class MeshletVisibility : u8
{
    Visible = 0,
    /// The bounding sphere is outside a frustum plane.
    FrustumCulled,
    /// Every triangle faces away from the camera.
    BackfaceCulled,
};

/// Tests one meshlet: sphere against each plane, then the cone test
/// `dot(center - eye, axis) >= cutoff * |center - eye| + radius`, which holds
/// only when every point of the sphere sees every normal in the cone from
/// behind.
[[nodiscard]] MeshletVisibility CullMeshlet(const MeshletBounds& bounds,
                                            const MeshletCullView& view) noexcept;

/// Counts from one reference cull.
struct MeshletCullStats
{
    u32 tested = 0;
    u32 frustumCulled = 0;
    u32 backfaceCulled = 0;
    u32 visible = 0;
    u32 visibleTriangles = 0;
};

/// The amplification stage on the CPU: the indices of surviving meshlets in
/// ascending order.
/// @return Success, or Internal on allocation failure.
[[nodiscard]] Status<Error> CullMeshletsReference(const MeshletMesh& mesh,
                                                  const MeshletCullView& view,
                                                  cyber::memory::DynamicArray<u32>& survivors,
                                                  MeshletCullStats* stats = nullptr) noexcept;

/// The vertex-pipeline fallback: expands meshlets into a triangle list of
/// source-vertex indices, in meshlet order.
/// @return Success, or InvalidArgument for a meshlet index out of range, or
/// Internal on allocation failure.
[[nodiscard]] Status<Error>
ExpandMeshletsToIndices(const MeshletMesh& mesh, cyber::memory::Span<const u32> meshletIndices,
                        cyber::memory::DynamicArray<u32>& indices) noexcept;

}  // namespace cyber::renderer
