#pragma once

// SPIR-V test shaders written with `SpirvBuilder`, shared by the Vulkan tests: the
// scale-and-bias compute shader, a full-screen triangle vertex shader, and two
// fragment shaders. Each has an HLSL twin in `backend_gpu_parity_test.cpp`.

#include "../src/vulkan/spirv_builder.h"

#include <cyber/memory/dynamic_array.h>

namespace cyber::rhi::test
{

using cyber::core::u32;
using cyber::memory::DynamicArray;
using cyber::rhi::detail::SpirvBuilder;
namespace spv = cyber::rhi::detail::spv;

inline constexpr u32 kFloatOne = 0x3F800000U;
inline constexpr u32 kFloatMinusOne = 0xBF800000U;
inline constexpr u32 kFloatThree = 0x40400000U;

// out[i] = in[i] * scale + bias for i < count. The shader every compute test
// dispatches: two storage buffers in set 0, three push constants.
//
//   layout(local_size_x = 64) in;
//   layout(set = 0, binding = 0) readonly buffer In  { uint data[]; } inb;
//   layout(set = 0, binding = 1)          buffer Out { uint data[]; } outb;
//   layout(push_constant) uniform PC { uint count, scale, bias; } pc;
//   void main() {
//       uint i = gl_GlobalInvocationID.x;
//       if (i < pc.count) outb.data[i] = inb.data[i] * pc.scale + pc.bias;
//   }
inline bool BuildScaleBiasCompute(SpirvBuilder& b, DynamicArray<u32>& out) noexcept
{
    using namespace spv;
    const u32 voidT = b.NewId(), fnT = b.NewId(), uintT = b.NewId(), v3uintT = b.NewId();
    const u32 ptrInV3 = b.NewId(), gid = b.NewId(), runtimeArray = b.NewId();
    const u32 inBlock = b.NewId(), outBlock = b.NewId();
    const u32 ptrSbIn = b.NewId(), ptrSbOut = b.NewId(), inVar = b.NewId(), outVar = b.NewId();
    const u32 pcStruct = b.NewId(), ptrPc = b.NewId(), pc = b.NewId();
    const u32 ptrPcUint = b.NewId(), ptrSbUint = b.NewId(), ptrInUint = b.NewId();
    const u32 c0 = b.NewId(), c1 = b.NewId(), c2 = b.NewId(), boolT = b.NewId();
    const u32 mainFn = b.NewId(), entry = b.NewId(), body = b.NewId(), merge = b.NewId();

    b.Capability(CapabilityShader);
    b.MemoryModelGlsl450();
    b.EntryPoint(ExecutionGLCompute, mainFn, "main", &gid, 1);
    b.LocalSize(mainFn, 64, 1, 1);
    b.Name(mainFn, "main");

    b.Decorate(gid, DecorationBuiltIn, BuiltInGlobalInvocationId);
    b.Decorate(runtimeArray, DecorationArrayStride, 4);
    b.Decorate(inBlock, DecorationBlock);
    b.MemberDecorate(inBlock, 0, DecorationOffset, 0);
    b.MemberDecorate(inBlock, 0, DecorationNonWritable);
    b.Decorate(outBlock, DecorationBlock);
    b.MemberDecorate(outBlock, 0, DecorationOffset, 0);
    b.Decorate(inVar, DecorationDescriptorSet, 0);
    b.Decorate(inVar, DecorationBinding, 0);
    b.Decorate(outVar, DecorationDescriptorSet, 0);
    b.Decorate(outVar, DecorationBinding, 1);
    b.Decorate(pcStruct, DecorationBlock);
    b.MemberDecorate(pcStruct, 0, DecorationOffset, 0);
    b.MemberDecorate(pcStruct, 1, DecorationOffset, 4);
    b.MemberDecorate(pcStruct, 2, DecorationOffset, 8);

    b.Declare(OpTypeVoid, {voidT});
    b.Declare(OpTypeFunction, {fnT, voidT});
    b.Declare(OpTypeInt, {uintT, 32, 0});
    b.Declare(OpTypeVector, {v3uintT, uintT, 3});
    b.Declare(OpTypePointer, {ptrInV3, StorageInput, v3uintT});
    b.Declare(OpVariable, {ptrInV3, gid, StorageInput});
    b.Declare(OpTypeRuntimeArray, {runtimeArray, uintT});
    b.Declare(OpTypeStruct, {inBlock, runtimeArray});
    b.Declare(OpTypeStruct, {outBlock, runtimeArray});
    b.Declare(OpTypePointer, {ptrSbIn, StorageStorageBuffer, inBlock});
    b.Declare(OpTypePointer, {ptrSbOut, StorageStorageBuffer, outBlock});
    b.Declare(OpVariable, {ptrSbIn, inVar, StorageStorageBuffer});
    b.Declare(OpVariable, {ptrSbOut, outVar, StorageStorageBuffer});
    b.Declare(OpTypeStruct, {pcStruct, uintT, uintT, uintT});
    b.Declare(OpTypePointer, {ptrPc, StoragePushConstant, pcStruct});
    b.Declare(OpVariable, {ptrPc, pc, StoragePushConstant});
    b.Declare(OpTypePointer, {ptrPcUint, StoragePushConstant, uintT});
    b.Declare(OpTypePointer, {ptrSbUint, StorageStorageBuffer, uintT});
    b.Declare(OpTypePointer, {ptrInUint, StorageInput, uintT});
    b.Declare(OpConstant, {uintT, c0, 0});
    b.Declare(OpConstant, {uintT, c1, 1});
    b.Declare(OpConstant, {uintT, c2, 2});
    b.Declare(OpTypeBool, {boolT});

    const u32 xPtr = b.NewId(), index = b.NewId(), countPtr = b.NewId(), count = b.NewId();
    const u32 cond = b.NewId(), scalePtr = b.NewId(), scale = b.NewId(), biasPtr = b.NewId();
    const u32 bias = b.NewId(), inPtr = b.NewId(), value = b.NewId(), scaled = b.NewId();
    const u32 result = b.NewId(), outPtr = b.NewId();

    b.Code(OpFunction, {voidT, mainFn, 0, fnT});
    b.Code(OpLabel, {entry});
    b.Code(OpAccessChain, {ptrInUint, xPtr, gid, c0});
    b.Code(OpLoad, {uintT, index, xPtr});
    b.Code(OpAccessChain, {ptrPcUint, countPtr, pc, c0});
    b.Code(OpLoad, {uintT, count, countPtr});
    b.Code(OpULessThan, {boolT, cond, index, count});
    b.Code(OpSelectionMerge, {merge, 0});
    b.Code(OpBranchConditional, {cond, body, merge});
    b.Code(OpLabel, {body});
    b.Code(OpAccessChain, {ptrPcUint, scalePtr, pc, c1});
    b.Code(OpLoad, {uintT, scale, scalePtr});
    b.Code(OpAccessChain, {ptrPcUint, biasPtr, pc, c2});
    b.Code(OpLoad, {uintT, bias, biasPtr});
    b.Code(OpAccessChain, {ptrSbUint, inPtr, inVar, c0, index});
    b.Code(OpLoad, {uintT, value, inPtr});
    b.Code(OpIMul, {uintT, scaled, value, scale});
    b.Code(OpIAdd, {uintT, result, scaled, bias});
    b.Code(OpAccessChain, {ptrSbUint, outPtr, outVar, c0, index});
    b.Code(OpStore, {outPtr, result});
    b.Code(OpBranch, {merge});
    b.Code(OpLabel, {merge});
    b.Code(OpReturn, {});
    b.Code(OpFunctionEnd, {});
    return b.Finish(out).IsOk();
}

// A full-screen triangle whose three corners are red, green and blue.
//
//   layout(location = 0) out vec3 color;
//   void main() {
//       gl_Position = vec4(gl_VertexIndex == 1 ? 3 : -1, gl_VertexIndex == 2 ? 3 : -1, 0, 1);
//       color = vec3(gl_VertexIndex == 0, gl_VertexIndex == 1, gl_VertexIndex == 2);
//   }
inline bool BuildTriangleVertex(SpirvBuilder& b, DynamicArray<u32>& out) noexcept
{
    using namespace spv;
    const u32 voidT = b.NewId(), fnT = b.NewId(), intT = b.NewId(), floatT = b.NewId();
    const u32 v3f = b.NewId(), v4f = b.NewId(), boolT = b.NewId();
    const u32 ptrInInt = b.NewId(), vertexIndex = b.NewId();
    const u32 ptrOutV4 = b.NewId(), position = b.NewId();
    const u32 ptrOutV3 = b.NewId(), colorOut = b.NewId();
    const u32 i0 = b.NewId(), i1 = b.NewId(), i2 = b.NewId();
    const u32 f0 = b.NewId(), f1 = b.NewId(), fMinus1 = b.NewId(), f3 = b.NewId();
    const u32 mainFn = b.NewId(), entry = b.NewId();

    b.Capability(CapabilityShader);
    b.MemoryModelGlsl450();
    const u32 interface[3] = {vertexIndex, position, colorOut};
    b.EntryPoint(ExecutionVertex, mainFn, "main", interface, 3);
    b.Decorate(vertexIndex, DecorationBuiltIn, BuiltInVertexIndex);
    b.Decorate(position, DecorationBuiltIn, BuiltInPosition);
    b.Decorate(colorOut, DecorationLocation, 0);

    b.Declare(OpTypeVoid, {voidT});
    b.Declare(OpTypeFunction, {fnT, voidT});
    b.Declare(OpTypeInt, {intT, 32, 1});
    b.Declare(OpTypeFloat, {floatT, 32});
    b.Declare(OpTypeVector, {v3f, floatT, 3});
    b.Declare(OpTypeVector, {v4f, floatT, 4});
    b.Declare(OpTypeBool, {boolT});
    b.Declare(OpTypePointer, {ptrInInt, StorageInput, intT});
    b.Declare(OpVariable, {ptrInInt, vertexIndex, StorageInput});
    b.Declare(OpTypePointer, {ptrOutV4, StorageOutput, v4f});
    b.Declare(OpVariable, {ptrOutV4, position, StorageOutput});
    b.Declare(OpTypePointer, {ptrOutV3, StorageOutput, v3f});
    b.Declare(OpVariable, {ptrOutV3, colorOut, StorageOutput});
    b.Declare(OpConstant, {intT, i0, 0});
    b.Declare(OpConstant, {intT, i1, 1});
    b.Declare(OpConstant, {intT, i2, 2});
    b.Declare(OpConstant, {floatT, f0, 0});
    b.Declare(OpConstant, {floatT, f1, kFloatOne});
    b.Declare(OpConstant, {floatT, fMinus1, kFloatMinusOne});
    b.Declare(OpConstant, {floatT, f3, kFloatThree});

    const u32 idx = b.NewId(), is0 = b.NewId(), is1 = b.NewId(), is2 = b.NewId();
    const u32 x = b.NewId(), y = b.NewId(), p = b.NewId(), r = b.NewId(), g = b.NewId();
    const u32 bl = b.NewId(), c = b.NewId();
    b.Code(OpFunction, {voidT, mainFn, 0, fnT});
    b.Code(OpLabel, {entry});
    b.Code(OpLoad, {intT, idx, vertexIndex});
    b.Code(OpIEqual, {boolT, is0, idx, i0});
    b.Code(OpIEqual, {boolT, is1, idx, i1});
    b.Code(OpIEqual, {boolT, is2, idx, i2});
    b.Code(OpSelect, {floatT, x, is1, f3, fMinus1});
    b.Code(OpSelect, {floatT, y, is2, f3, fMinus1});
    b.Code(OpCompositeConstruct, {v4f, p, x, y, f0, f1});
    b.Code(OpStore, {position, p});
    b.Code(OpSelect, {floatT, r, is0, f1, f0});
    b.Code(OpSelect, {floatT, g, is1, f1, f0});
    b.Code(OpSelect, {floatT, bl, is2, f1, f0});
    b.Code(OpCompositeConstruct, {v3f, c, r, g, bl});
    b.Code(OpStore, {colorOut, c});
    b.Code(OpReturn, {});
    b.Code(OpFunctionEnd, {});
    return b.Finish(out).IsOk();
}

// Writes the interpolated colour times a push-constant brightness.
//
//   layout(location = 0) in vec3 color;
//   layout(location = 0) out vec4 outColor;
//   layout(push_constant) uniform PC { float brightness; } pc;
//   void main() { outColor = vec4(color * pc.brightness, 1.0); }
inline bool BuildColorFragment(SpirvBuilder& b, DynamicArray<u32>& out) noexcept
{
    using namespace spv;
    const u32 voidT = b.NewId(), fnT = b.NewId(), floatT = b.NewId(), uintT = b.NewId();
    const u32 v3f = b.NewId(), v4f = b.NewId();
    const u32 ptrInV3 = b.NewId(), colorIn = b.NewId(), ptrOutV4 = b.NewId(), outColor = b.NewId();
    const u32 pcStruct = b.NewId(), ptrPc = b.NewId(), pc = b.NewId(), ptrPcFloat = b.NewId();
    const u32 c0 = b.NewId(), f1 = b.NewId(), mainFn = b.NewId(), entry = b.NewId();

    b.Capability(CapabilityShader);
    b.MemoryModelGlsl450();
    const u32 interface[2] = {colorIn, outColor};
    b.EntryPoint(ExecutionFragment, mainFn, "main", interface, 2);
    b.OriginUpperLeft(mainFn);
    b.Decorate(colorIn, DecorationLocation, 0);
    b.Decorate(outColor, DecorationLocation, 0);
    b.Decorate(pcStruct, DecorationBlock);
    b.MemberDecorate(pcStruct, 0, DecorationOffset, 0);

    b.Declare(OpTypeVoid, {voidT});
    b.Declare(OpTypeFunction, {fnT, voidT});
    b.Declare(OpTypeFloat, {floatT, 32});
    b.Declare(OpTypeInt, {uintT, 32, 0});
    b.Declare(OpTypeVector, {v3f, floatT, 3});
    b.Declare(OpTypeVector, {v4f, floatT, 4});
    b.Declare(OpTypePointer, {ptrInV3, StorageInput, v3f});
    b.Declare(OpVariable, {ptrInV3, colorIn, StorageInput});
    b.Declare(OpTypePointer, {ptrOutV4, StorageOutput, v4f});
    b.Declare(OpVariable, {ptrOutV4, outColor, StorageOutput});
    b.Declare(OpTypeStruct, {pcStruct, floatT});
    b.Declare(OpTypePointer, {ptrPc, StoragePushConstant, pcStruct});
    b.Declare(OpVariable, {ptrPc, pc, StoragePushConstant});
    b.Declare(OpTypePointer, {ptrPcFloat, StoragePushConstant, floatT});
    b.Declare(OpConstant, {uintT, c0, 0});
    b.Declare(OpConstant, {floatT, f1, kFloatOne});

    const u32 color = b.NewId(), scalePtr = b.NewId(), scale = b.NewId(), scaled = b.NewId();
    const u32 r = b.NewId(), g = b.NewId(), bl = b.NewId(), result = b.NewId();
    b.Code(OpFunction, {voidT, mainFn, 0, fnT});
    b.Code(OpLabel, {entry});
    b.Code(OpLoad, {v3f, color, colorIn});
    b.Code(OpAccessChain, {ptrPcFloat, scalePtr, pc, c0});
    b.Code(OpLoad, {floatT, scale, scalePtr});
    b.Code(OpVectorTimesScalar, {v3f, scaled, color, scale});
    b.Code(OpCompositeExtract, {floatT, r, scaled, 0});
    b.Code(OpCompositeExtract, {floatT, g, scaled, 1});
    b.Code(OpCompositeExtract, {floatT, bl, scaled, 2});
    b.Code(OpCompositeConstruct, {v4f, result, r, g, bl, f1});
    b.Code(OpStore, {outColor, result});
    b.Code(OpReturn, {});
    b.Code(OpFunctionEnd, {});
    return b.Finish(out).IsOk();
}


// Writes the interpolated colour with alpha one and no push constants.
//
//   layout(location = 0) in vec3 color;
//   layout(location = 0) out vec4 outColor;
//   void main() { outColor = vec4(color, 1.0); }
inline bool BuildPlainColorFragment(SpirvBuilder& b, DynamicArray<u32>& out) noexcept
{
    using namespace cyber::rhi::detail::spv;
    const u32 voidT = b.NewId(), fnT = b.NewId(), floatT = b.NewId();
    const u32 v3f = b.NewId(), v4f = b.NewId();
    const u32 ptrInV3 = b.NewId(), colorIn = b.NewId(), ptrOutV4 = b.NewId(), outColor = b.NewId();
    const u32 f1 = b.NewId(), mainFn = b.NewId(), entry = b.NewId();

    b.Capability(CapabilityShader);
    b.MemoryModelGlsl450();
    const u32 interface[2] = {colorIn, outColor};
    b.EntryPoint(ExecutionFragment, mainFn, "main", interface, 2);
    b.OriginUpperLeft(mainFn);
    b.Decorate(colorIn, DecorationLocation, 0);
    b.Decorate(outColor, DecorationLocation, 0);

    b.Declare(OpTypeVoid, {voidT});
    b.Declare(OpTypeFunction, {fnT, voidT});
    b.Declare(OpTypeFloat, {floatT, 32});
    b.Declare(OpTypeVector, {v3f, floatT, 3});
    b.Declare(OpTypeVector, {v4f, floatT, 4});
    b.Declare(OpTypePointer, {ptrInV3, StorageInput, v3f});
    b.Declare(OpVariable, {ptrInV3, colorIn, StorageInput});
    b.Declare(OpTypePointer, {ptrOutV4, StorageOutput, v4f});
    b.Declare(OpVariable, {ptrOutV4, outColor, StorageOutput});
    b.Declare(OpConstant, {floatT, f1, kFloatOne});

    const u32 color = b.NewId(), r = b.NewId(), g = b.NewId(), bl = b.NewId(), result = b.NewId();
    b.Code(OpFunction, {voidT, mainFn, 0, fnT});
    b.Code(OpLabel, {entry});
    b.Code(OpLoad, {v3f, color, colorIn});
    b.Code(OpCompositeExtract, {floatT, r, color, 0});
    b.Code(OpCompositeExtract, {floatT, g, color, 1});
    b.Code(OpCompositeExtract, {floatT, bl, color, 2});
    b.Code(OpCompositeConstruct, {v4f, result, r, g, bl, f1});
    b.Code(OpStore, {outColor, result});
    b.Code(OpReturn, {});
    b.Code(OpFunctionEnd, {});
    return b.Finish(out).IsOk();
}

}  // namespace cyber::rhi::test
