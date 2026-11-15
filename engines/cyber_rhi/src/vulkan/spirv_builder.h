#pragma once

// A SPIR-V module writer: the emission half of the SPIR-V generator Phase 34 needs.
// ADR 0019 chose the project's own shader compiler; the Vulkan backend consumes
// SPIR-V, so this is the layer a code generator writes into. It knows the module
// layout the specification requires (capabilities, memory model, entry points,
// execution modes, debug, annotations, types and globals, functions) and keeps
// each section in order however the caller interleaves emission. It validates
// nothing about types; correctness of the instruction stream is the generator's.

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/dynamic_array.h>

#include <cstring>
#include <initializer_list>

namespace cyber::rhi::detail
{

namespace spv
{
using cyber::core::u32;

inline constexpr u32 kMagic = 0x07230203U;
inline constexpr u32 kVersion13 = 0x00010300U;

// Opcodes the backend's generators use.
enum Op : u32
{
    OpName = 5,
    OpMemoryModel = 14,
    OpEntryPoint = 15,
    OpExecutionMode = 16,
    OpCapability = 17,
    OpTypeVoid = 19,
    OpTypeBool = 20,
    OpTypeInt = 21,
    OpTypeFloat = 22,
    OpTypeVector = 23,
    OpTypeArray = 28,
    OpTypeRuntimeArray = 29,
    OpTypeStruct = 30,
    OpTypePointer = 32,
    OpTypeFunction = 33,
    OpConstant = 43,
    OpConstantComposite = 44,
    OpFunction = 54,
    OpFunctionEnd = 56,
    OpVariable = 59,
    OpLoad = 61,
    OpStore = 62,
    OpAccessChain = 65,
    OpDecorate = 71,
    OpMemberDecorate = 72,
    OpCompositeConstruct = 80,
    OpCompositeExtract = 81,
    OpIAdd = 128,
    OpFMul = 133,
    OpIMul = 132,
    OpVectorTimesScalar = 142,
    OpSelect = 169,
    OpIEqual = 170,
    OpULessThan = 176,
    OpSelectionMerge = 247,
    OpLabel = 248,
    OpBranch = 249,
    OpBranchConditional = 250,
    OpReturn = 253,
};

enum Capability : u32
{
    CapabilityShader = 1,
};
enum AddressingModel : u32
{
    AddressingLogical = 0,
};
enum MemoryModel : u32
{
    MemoryGLSL450 = 1,
};
enum ExecutionModel : u32
{
    ExecutionVertex = 0,
    ExecutionFragment = 4,
    ExecutionGLCompute = 5,
};
enum ExecutionMode : u32
{
    ModeOriginUpperLeft = 7,
    ModeLocalSize = 17,
};
enum StorageClass : u32
{
    StorageUniformConstant = 0,
    StorageInput = 1,
    StorageUniform = 2,
    StorageOutput = 3,
    StoragePushConstant = 9,
    StorageStorageBuffer = 12,
};
enum Decoration : u32
{
    DecorationBlock = 2,
    DecorationArrayStride = 6,
    DecorationBuiltIn = 11,
    DecorationNonWritable = 24,
    DecorationLocation = 30,
    DecorationBinding = 33,
    DecorationDescriptorSet = 34,
    DecorationOffset = 35,
};
enum BuiltIn : u32
{
    BuiltInPosition = 0,
    BuiltInVertexIndex = 42,
    BuiltInGlobalInvocationId = 28,
};
}  // namespace spv

class SpirvBuilder
{
public:
    using u32 = cyber::core::u32;
    using usize = cyber::core::usize;

    explicit SpirvBuilder(memory::Allocator& allocator) noexcept
        : m_capabilities(allocator), m_entryPoints(allocator), m_debug(allocator),
          m_annotations(allocator), m_declarations(allocator), m_code(allocator)
    {
    }

    // A fresh result id.
    [[nodiscard]] u32 NewId() noexcept
    {
        return m_bound++;
    }

    void Capability(spv::Capability capability) noexcept
    {
        Emit(m_capabilities, spv::OpCapability, {capability});
    }

    // One memory model per module; the generators here always use GLSL450.
    void MemoryModelGlsl450() noexcept
    {
        Emit(m_entryPoints, spv::OpMemoryModel, {spv::AddressingLogical, spv::MemoryGLSL450});
    }

    void EntryPoint(spv::ExecutionModel model, u32 function, const char* name, const u32* interface,
                    u32 interfaceCount) noexcept
    {
        EmitWithString(m_entryPoints, spv::OpEntryPoint, {model, function}, name, interface,
                       interfaceCount);
    }

    void LocalSize(u32 function, u32 x, u32 y, u32 z) noexcept
    {
        Emit(m_entryPoints, spv::OpExecutionMode, {function, spv::ModeLocalSize, x, y, z});
    }

    void OriginUpperLeft(u32 function) noexcept
    {
        Emit(m_entryPoints, spv::OpExecutionMode, {function, spv::ModeOriginUpperLeft});
    }

    void Name(u32 target, const char* name) noexcept
    {
        EmitWithString(m_debug, spv::OpName, {target}, name, nullptr, 0);
    }

    void Decorate(u32 target, spv::Decoration decoration, u32 argument = ~0U) noexcept
    {
        if (argument == ~0U)
        {
            Emit(m_annotations, spv::OpDecorate, {target, decoration});
        }
        else
        {
            Emit(m_annotations, spv::OpDecorate, {target, decoration, argument});
        }
    }

    void MemberDecorate(u32 structType, u32 member, spv::Decoration decoration,
                        u32 argument = ~0U) noexcept
    {
        if (argument == ~0U)
        {
            Emit(m_annotations, spv::OpMemberDecorate, {structType, member, decoration});
        }
        else
        {
            Emit(m_annotations, spv::OpMemberDecorate, {structType, member, decoration, argument});
        }
    }

    // Types, constants and global variables share one section and must appear
    // after anything they reference.
    void Declare(spv::Op op, std::initializer_list<u32> operands) noexcept
    {
        Emit(m_declarations, op, operands);
    }

    // Function bodies and their instructions.
    void Code(spv::Op op, std::initializer_list<u32> operands) noexcept
    {
        Emit(m_code, op, operands);
    }

    // Assembles the module. `ErrorCode::Internal` when an allocation failed while
    // emitting.
    [[nodiscard]] cyber::core::Status<cyber::core::Error>
    Finish(cyber::memory::DynamicArray<u32>& out) const noexcept
    {
        using Status = cyber::core::Status<cyber::core::Error>;
        if (m_failed)
        {
            return Status::Err(cyber::core::Error{cyber::core::ErrorCode::Internal,
                                                  "SPIR-V emission ran out of memory"});
        }
        out.Clear();
        const DynamicArray* sections[] = {&m_capabilities, &m_entryPoints,  &m_debug,
                                          &m_annotations,  &m_declarations, &m_code};
        usize total = 5;
        for (const DynamicArray* section : sections)
        {
            total += section->Size();
        }
        if (!out.TryReserve(total))
        {
            return Status::Err(cyber::core::Error{cyber::core::ErrorCode::Internal,
                                                  "SPIR-V assembly ran out of memory"});
        }
        const u32 header[5] = {spv::kMagic, spv::kVersion13, 0, m_bound, 0};
        for (const u32 word : header)
        {
            if (!out.TryPushBack(word))
            {
                return Status::Err(cyber::core::Error{cyber::core::ErrorCode::Internal,
                                                      "SPIR-V assembly ran out of memory"});
            }
        }
        for (const DynamicArray* section : sections)
        {
            for (usize index = 0; index < section->Size(); ++index)
            {
                if (!out.TryPushBack((*section)[index]))
                {
                    return Status::Err(cyber::core::Error{cyber::core::ErrorCode::Internal,
                                                          "SPIR-V assembly ran out of memory"});
                }
            }
        }
        return Status::Ok();
    }

private:
    using DynamicArray = cyber::memory::DynamicArray<u32>;

    void Push(DynamicArray& section, u32 word) noexcept
    {
        if (!section.TryPushBack(word))
        {
            m_failed = true;
        }
    }

    void Emit(DynamicArray& section, u32 opcode, std::initializer_list<u32> operands) noexcept
    {
        Push(section, (static_cast<u32>(operands.size() + 1) << 16U) | opcode);
        for (const u32 operand : operands)
        {
            Push(section, operand);
        }
    }

    // Instructions that carry a nul-terminated string packed little-endian into
    // words, followed by optional trailing ids (an entry point's interface).
    void EmitWithString(DynamicArray& section, u32 opcode, std::initializer_list<u32> head,
                        const char* text, const u32* tail, u32 tailCount) noexcept
    {
        const usize length = std::strlen(text) + 1;
        const u32 stringWords = static_cast<u32>((length + 3) / 4);
        const u32 wordCount = 1 + static_cast<u32>(head.size()) + stringWords + tailCount;
        Push(section, (wordCount << 16U) | opcode);
        for (const u32 operand : head)
        {
            Push(section, operand);
        }
        for (u32 word = 0; word < stringWords; ++word)
        {
            u32 packed = 0;
            for (u32 byte = 0; byte < 4; ++byte)
            {
                const usize index = static_cast<usize>(word) * 4 + byte;
                const u32 value = index < length ? static_cast<unsigned char>(text[index]) : 0U;
                packed |= value << (8U * byte);
            }
            Push(section, packed);
        }
        for (u32 index = 0; index < tailCount; ++index)
        {
            Push(section, tail[index]);
        }
    }

    DynamicArray m_capabilities;
    DynamicArray m_entryPoints;
    DynamicArray m_debug;
    DynamicArray m_annotations;
    DynamicArray m_declarations;
    DynamicArray m_code;
    u32 m_bound = 1;
    bool m_failed = false;
};

}  // namespace cyber::rhi::detail
