#pragma once

// The Vulkan backend's HLSL compiler: HLSL source in, SPIR-V 1.5 out.
//
// The D3D12 backend accepts `ShaderBytecodeFormat::HlslSource` and compiles it with
// the compiler Windows ships. This is the same contract for Vulkan, written in the
// project (ADR 0019, ADR 0029): the renderer's HLSL, and the HLSL the Cyber shader
// language generates, runs on Vulkan unchanged. It is a compiler for the subset of
// HLSL (shader model 5.1 style) the engine uses, and it refuses anything outside
// that subset by name with a source location instead of guessing.
//
// The translation follows the conventions that make D3D12-shaped code behave the
// same on both backends:
//   * `register(<class><n>, space<s>)` is resolved against the pipeline layout the
//     shader is used with, with the register assignment the D3D12 root signature
//     uses (`BindingResolver`); `b0, space0` is the push-constant block when the
//     layout has push constants, as D3D12 maps root constants there;
//   * HLSL matrices keep their row/column meaning: an HLSL `floatRxC` is a SPIR-V
//     matrix whose columns are the HLSL rows, so `m[i]`, constructors and `mul`
//     need no transposes, and the default column-major packing becomes `RowMajor`;
//   * constant buffers use HLSL packing (16-byte registers, no straddling) and
//     structured buffers HLSL's tight packing, both as explicit offsets;
//   * `&&`, `||` and `?:` evaluate both operands as FXC does;
//   * vertex outputs and pixel inputs are linked by semantic, not declaration order;
//   * `SV_Position.w` in a pixel shader is clip w, `SV_InstanceID` starts at 0.

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/dynamic_array.h>
#include <cyber/rhi/shader.h>

namespace cyber::rhi::detail::hlsl
{

using cyber::core::u32;
using cyber::core::u8;
using cyber::core::usize;

// HLSL register classes.
enum class RegisterClass : u8
{
    ConstantBuffer = 0,  // b
    ShaderResource,      // t
    UnorderedAccess,     // u
    Sampler,             // s
};

// What a resource binds to, as the binding resolver answers it.
enum class BindingKind : u8
{
    UniformBuffer = 0,
    StorageBuffer,
    SampledImage,
    StorageImage,
    Sampler,
    PushConstant,
};

struct ResolvedBinding
{
    BindingKind kind = BindingKind::UniformBuffer;
    u32 set = 0;
    u32 binding = 0;
    // Descriptors in the binding. 0 means the binding is a runtime-sized array.
    u32 arrayCount = 1;
    // Element of the binding the register names (register minus the binding's base).
    u32 firstElement = 0;
};

// Maps an HLSL register to a descriptor. Returns false when the layout has no
// binding for it; the compiler then fails with the register named.
class BindingResolver
{
public:
    virtual ~BindingResolver() = default;
    [[nodiscard]] virtual bool Resolve(RegisterClass registerClass, u32 registerIndex, u32 space,
                                       ResolvedBinding* out) const noexcept = 0;
};

// One linked stage varying: a non-system semantic and the location it occupies.
struct Varying
{
    char semantic[32] = {};  // upper case, without the trailing index
    u32 index = 0;
    u32 location = 0;
};

inline constexpr u32 kMaxVaryings = 32;

// Device features the generated code may rely on.
struct CompileFeatures
{
    // `SV_InstanceID` is computed as InstanceIndex - BaseInstance; a shader that
    // reads it is refused when this is false.
    bool drawParameters = false;
    bool storageImageReadWithoutFormat = false;
    bool storageImageWriteWithoutFormat = false;
};

struct CompileDesc
{
    const char* source = nullptr;
    usize sourceBytes = 0;
    const char* entryPoint = "main";
    ShaderStage stage = ShaderStage::Vertex;
    // Optional diagnostic name (the shader label).
    const char* sourceName = nullptr;
    // Null: `set = space`, `binding = register` plus 0, 1000, 2000 or 3000 for the
    // b, t, u and s classes, kinds from the declaration, and `b0, space0` is a
    // uniform buffer. Used to check a shader before it has a layout.
    const BindingResolver* resolver = nullptr;
    // Pixel stage only: the vertex stage's outputs. Null assigns locations in
    // declaration order, which is what the vertex stage does for its outputs.
    const Varying* linkedInputs = nullptr;
    u32 linkedInputCount = 0;
    CompileFeatures features{};
};

struct CompileOutput
{
    explicit CompileOutput(cyber::memory::Allocator& allocator) noexcept : words(allocator) {}

    cyber::memory::DynamicArray<u32> words;
    // Vertex stage: the varyings it writes, for linking the pixel stage.
    Varying outputs[kMaxVaryings] = {};
    u32 outputCount = 0;
    u32 localSize[3] = {1, 1, 1};
    // The first error: "name:line:col: message". Empty on success.
    char diagnostic[320] = {};
};

// Compiles `desc.entryPoint` of `desc.source` for `desc.stage`. Scratch memory comes
// from `allocator` and is released before returning. On failure the error message
// points at `out->diagnostic`, so it lives as long as `out`.
[[nodiscard]] cyber::core::Status<cyber::core::Error>
Compile(const CompileDesc& desc, cyber::memory::Allocator& allocator, CompileOutput* out) noexcept;

}  // namespace cyber::rhi::detail::hlsl
