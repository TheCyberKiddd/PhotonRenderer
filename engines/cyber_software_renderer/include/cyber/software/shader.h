#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/math/mat4.h>
#include <cyber/math/vec3.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/pipeline.h>
#include <cyber/software/shading.h>

namespace cyber::jobs
{
class JobSystem;
}

namespace cyber::software
{

class DepthBuffer;

// User data for `TransformVertexShader`: the matrices a scene applies before
// the pipeline clips and rasterizes.
struct TransformShaderData
{
    // Model -> view -> clip. Usually `projection * view * model`.
    math::Mat4 modelViewProjection = math::Mat4::Identity();
    // Model -> world, used for the interpolated world position.
    math::Mat4 modelToWorld = math::Mat4::Identity();
    // Inverse-transpose of the upper-left 3x3, used for normals.
    math::Mat4 normalToWorld = math::Mat4::Identity();
    // World-space eye position, copied into the varyings for lighting.
    math::Vec3 eyePosition;
};

// Built-in vertex shader. Transforms position into clip space, world position
// through `modelToWorld`, and the normal through `normalToWorld`, and forwards
// UV and vertex color. Requires `userData` to point at a `TransformShaderData`;
// a null pointer produces an origin vertex, which the pipeline then culls.
void TransformVertexShader(const VertexInput& input, const void* userData,
                           VertexStageOutput& output) noexcept;

struct ShaderDesc
{
    PipelineDesc pipeline{};
    // Required. A null shader returns InvalidArgument.
    VertexShaderFn vertexShader = nullptr;
    // Required. A null shader returns InvalidArgument.
    PixelShaderFn pixelShader = nullptr;
    // Passed unchanged to both stages. May be null when the shaders do not need
    // it.
    const void* userData = nullptr;
};

// Depth-only vertex stage. Returns the homogeneous clip position for one
// object-space vertex and produces no varyings, because the depth-only
// pipeline interpolates none. Must be noexcept and must not allocate.
using DepthOnlyVertexShaderFn = math::Vec4 (*)(const VertexInput& input,
                                               const void* userData) noexcept;

// Built-in depth-only vertex shader. Transforms the object-space position by
// `clipFromObject`. Requires `userData` to point at a `math::Mat4`; a null
// pointer produces an origin vertex, which the pipeline then culls.
[[nodiscard]] math::Vec4 TransformDepthOnlyVertexShader(const VertexInput& input,
                                                        const void* userData) noexcept;

struct DepthOnlyShaderDesc
{
    PipelineDesc pipeline{};
    // Required. A null shader returns InvalidArgument.
    DepthOnlyVertexShaderFn vertexShader = nullptr;
    // Passed unchanged to the vertex stage. May be null when the shader does
    // not need it.
    const void* userData = nullptr;
};

// Runs the vertex pipeline over a triangle list and writes depth only.
//
// For every three vertices: run `desc.vertexShader`, clip the near plane in
// homogeneous space, divide by w, map into the viewport with a y flip, cull by
// `desc.pipeline.cullMode`/`frontFace`, and reject triangles whose screen
// coordinates leave the +/-16384-pixel guard band. Surviving triangles are
// rasterized into `depth` with the reverse-Z greater-or-equal test. No
// varyings are carried through the clipper, no pixel shader runs, and no color
// target exists, so this is the shadow-map and depth-prepass path. Coverage
// and the resulting depth values match `DrawShadedTriangles` for the same
// geometry; alpha-tested or shader-discarded occluders are not supported,
// because there is no pixel stage.
//
// The caller clears `depth` before the draw. The allocator supplies temporary
// clip and triangle arrays and is not accessed by workers.
//
// Returns InvalidArgument for an invalid depth buffer, an invalid viewport, a
// null shader, or a vertex count that is not a multiple of three; Overflow when
// a temporary count exceeds 32 bits; Internal when an allocation fails.
// `stats` is optional and is zeroed on entry.
[[nodiscard]] core::Status<core::Error>
DrawDepthOnlyTriangles(const DepthOnlyShaderDesc& desc, memory::Span<const VertexInput> vertices,
                       DepthBuffer& depth, jobs::JobSystem* jobSystem,
                       memory::Allocator& allocator, PipelineStats* stats = nullptr) noexcept;

// Runs the Phase 9 vertex pipeline over a triangle list.
//
// For every three vertices: run `desc.vertexShader`, clip the near plane in
// homogeneous space (interpolating every varying), divide by w, map into the
// viewport with a y flip, cull by `desc.pipeline.cullMode`/`frontFace`, and
// reject triangles whose screen coordinates leave the +/-16384-pixel guard
// band. Surviving triangles are rasterized into `color` and `depth` with a
// reverse-Z depth test, perspective-correct varying interpolation, and
// `desc.pixelShader` producing each pixel.
//
// `depth` must be valid and have the color target's dimensions. The caller
// clears both targets before the draw. The allocator supplies temporary clip
// and triangle arrays and is not accessed by workers; the pixel shader runs on
// the thread that owns the tile.
//
// Returns InvalidArgument for an invalid target, a dimension mismatch, an
// invalid viewport, a null shader, or a vertex count that is not a multiple of
// three; Overflow when a temporary count exceeds 32 bits; Internal when an
// allocation fails. `stats` is optional and is zeroed on entry.
[[nodiscard]] core::Status<core::Error>
DrawShadedTriangles(const ShaderDesc& desc, memory::Span<const VertexInput> vertices,
                    Framebuffer& color, DepthBuffer& depth, jobs::JobSystem* jobSystem,
                    memory::Allocator& allocator, PipelineStats* stats = nullptr) noexcept;

}  // namespace cyber::software
