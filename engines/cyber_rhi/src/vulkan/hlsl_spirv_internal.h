#pragma once

// Private state of the HLSL to SPIR-V compiler (hlsl_spirv.h). One `Compiler` per
// compilation; every allocation comes from an arena over the caller's allocator
// and is released when the compiler is destroyed.
//
// The compiler is a single pass over a token array: declarations are recorded,
// function bodies are compiled on first call (so code the entry point never
// reaches, and the resources only it names, are never emitted), and expressions
// emit SPIR-V as they are parsed. A `for` loop's increment is compiled by
// replaying its tokens after the body, where SPIR-V wants the continue block.

#include "hlsl_spirv.h"

#include <cstring>
#include <initializer_list>

namespace cyber::rhi::detail::hlsl
{

using cyber::core::u16;
using cyber::core::u64;

namespace spv
{
enum Op : u32
{
    OpUndef = 1,
    OpName = 5,
    OpMemberName = 6,
    OpExtInstImport = 11,
    OpExtInst = 12,
    OpMemoryModel = 14,
    OpEntryPoint = 15,
    OpExecutionMode = 16,
    OpCapability = 17,
    OpTypeVoid = 19,
    OpTypeBool = 20,
    OpTypeInt = 21,
    OpTypeFloat = 22,
    OpTypeVector = 23,
    OpTypeMatrix = 24,
    OpTypeImage = 25,
    OpTypeSampler = 26,
    OpTypeSampledImage = 27,
    OpTypeArray = 28,
    OpTypeRuntimeArray = 29,
    OpTypeStruct = 30,
    OpTypePointer = 32,
    OpTypeFunction = 33,
    OpConstantTrue = 41,
    OpConstantFalse = 42,
    OpConstant = 43,
    OpConstantComposite = 44,
    OpConstantNull = 46,
    OpFunction = 54,
    OpFunctionParameter = 55,
    OpFunctionEnd = 56,
    OpFunctionCall = 57,
    OpVariable = 59,
    OpLoad = 61,
    OpStore = 62,
    OpAccessChain = 65,
    OpArrayLength = 68,
    OpDecorate = 71,
    OpMemberDecorate = 72,
    OpVectorExtractDynamic = 77,
    OpVectorShuffle = 79,
    OpCompositeConstruct = 80,
    OpCompositeExtract = 81,
    OpCompositeInsert = 82,
    OpTranspose = 84,
    OpSampledImage = 86,
    OpImageSampleImplicitLod = 87,
    OpImageSampleExplicitLod = 88,
    OpImageSampleDrefImplicitLod = 89,
    OpImageSampleDrefExplicitLod = 90,
    OpImageFetch = 95,
    OpImageGather = 96,
    OpImageRead = 98,
    OpImageWrite = 99,
    OpImageQuerySizeLod = 103,
    OpImageQuerySize = 104,
    OpImageQueryLevels = 106,
    OpConvertFToU = 109,
    OpConvertFToS = 110,
    OpConvertSToF = 111,
    OpConvertUToF = 112,
    OpBitcast = 124,
    OpSNegate = 126,
    OpFNegate = 127,
    OpIAdd = 128,
    OpFAdd = 129,
    OpISub = 130,
    OpFSub = 131,
    OpIMul = 132,
    OpFMul = 133,
    OpUDiv = 134,
    OpSDiv = 135,
    OpFDiv = 136,
    OpUMod = 137,
    OpSRem = 138,
    OpFRem = 140,
    OpVectorTimesScalar = 142,
    OpMatrixTimesScalar = 143,
    OpVectorTimesMatrix = 144,
    OpMatrixTimesVector = 145,
    OpMatrixTimesMatrix = 146,
    OpDot = 148,
    OpAny = 154,
    OpAll = 155,
    OpIsNan = 156,
    OpIsInf = 157,
    OpLogicalEqual = 164,
    OpLogicalNotEqual = 165,
    OpLogicalOr = 166,
    OpLogicalAnd = 167,
    OpLogicalNot = 168,
    OpSelect = 169,
    OpIEqual = 170,
    OpINotEqual = 171,
    OpUGreaterThan = 172,
    OpSGreaterThan = 173,
    OpUGreaterThanEqual = 174,
    OpSGreaterThanEqual = 175,
    OpULessThan = 176,
    OpSLessThan = 177,
    OpULessThanEqual = 178,
    OpSLessThanEqual = 179,
    OpFOrdEqual = 180,
    OpFUnordNotEqual = 183,
    OpFOrdLessThan = 184,
    OpFOrdGreaterThan = 186,
    OpFOrdLessThanEqual = 188,
    OpFOrdGreaterThanEqual = 190,
    OpShiftRightLogical = 194,
    OpShiftRightArithmetic = 195,
    OpShiftLeftLogical = 196,
    OpBitwiseOr = 197,
    OpBitwiseXor = 198,
    OpBitwiseAnd = 199,
    OpNot = 200,
    OpBitReverse = 204,
    OpBitCount = 205,
    OpDPdx = 207,
    OpDPdy = 208,
    OpFwidth = 209,
    OpControlBarrier = 224,
    OpMemoryBarrier = 225,
    OpAtomicExchange = 229,
    OpAtomicCompareExchange = 230,
    OpAtomicIAdd = 234,
    OpAtomicSMin = 236,
    OpAtomicUMin = 237,
    OpAtomicSMax = 238,
    OpAtomicUMax = 239,
    OpAtomicAnd = 240,
    OpAtomicOr = 241,
    OpAtomicXor = 242,
    OpLoopMerge = 246,
    OpSelectionMerge = 247,
    OpLabel = 248,
    OpBranch = 249,
    OpBranchConditional = 250,
    OpKill = 252,
    OpReturn = 253,
    OpReturnValue = 254,
    OpUnreachable = 255,
};

enum StorageClass : u32
{
    UniformConstant = 0,
    Input = 1,
    Uniform = 2,
    Output = 3,
    Workgroup = 4,
    Private = 6,
    Function = 7,
    PushConstant = 9,
    StorageBuffer = 12,
};

enum Decoration : u32
{
    DecBlock = 2,
    DecRowMajor = 4,
    DecArrayStride = 6,
    DecMatrixStride = 7,
    DecBuiltIn = 11,
    DecNoPerspective = 13,
    DecFlat = 14,
    DecCentroid = 16,
    DecSample = 17,
    DecNonWritable = 24,
    DecLocation = 30,
    DecBinding = 33,
    DecDescriptorSet = 34,
    DecOffset = 35,
    DecNoContraction = 42,
    DecNonUniform = 5300,
};

enum BuiltIn : u32
{
    BuiltInPosition = 0,
    BuiltInPrimitiveId = 7,
    BuiltInFragCoord = 15,
    BuiltInFrontFacing = 17,
    BuiltInSampleId = 18,
    BuiltInFragDepth = 22,
    BuiltInWorkgroupId = 26,
    BuiltInLocalInvocationId = 27,
    BuiltInGlobalInvocationId = 28,
    BuiltInLocalInvocationIndex = 29,
    BuiltInVertexIndex = 42,
    BuiltInInstanceIndex = 43,
    BuiltInBaseInstance = 4425,
};

enum Capability : u32
{
    CapMatrix = 0,
    CapShader = 1,
    CapImageGatherExtended = 25,
    CapImageCubeArray = 34,
    CapSampled1D = 43,
    CapImage1D = 44,
    CapSampledCubeArray = 45,
    CapImageQuery = 50,
    CapDerivativeControl = 51,
    CapStorageImageReadWithoutFormat = 55,
    CapStorageImageWriteWithoutFormat = 56,
    CapDrawParameters = 4427,
    CapShaderNonUniform = 5301,
    CapRuntimeDescriptorArray = 5302,
    CapUniformBufferArrayNonUniformIndexing = 5306,
    CapSampledImageArrayNonUniformIndexing = 5307,
    CapStorageBufferArrayNonUniformIndexing = 5308,
    CapStorageImageArrayNonUniformIndexing = 5309,
};

enum Glsl : u32
{
    GlslRoundEven = 2,
    GlslTrunc = 3,
    GlslFAbs = 4,
    GlslSAbs = 5,
    GlslFSign = 6,
    GlslSSign = 7,
    GlslFloor = 8,
    GlslCeil = 9,
    GlslFract = 10,
    GlslRadians = 11,
    GlslDegrees = 12,
    GlslSin = 13,
    GlslCos = 14,
    GlslTan = 15,
    GlslAsin = 16,
    GlslAcos = 17,
    GlslAtan = 18,
    GlslSinh = 19,
    GlslCosh = 20,
    GlslTanh = 21,
    GlslAtan2 = 25,
    GlslPow = 26,
    GlslExp = 27,
    GlslLog = 28,
    GlslExp2 = 29,
    GlslLog2 = 30,
    GlslSqrt = 31,
    GlslInverseSqrt = 32,
    GlslDeterminant = 33,
    GlslFMin = 37,
    GlslUMin = 38,
    GlslSMin = 39,
    GlslFMax = 40,
    GlslUMax = 41,
    GlslSMax = 42,
    GlslFClamp = 43,
    GlslUClamp = 44,
    GlslSClamp = 45,
    GlslFMix = 46,
    GlslStep = 48,
    GlslSmoothStep = 49,
    GlslFma = 50,
    GlslLdexp = 53,
    GlslPackHalf2x16 = 58,
    GlslUnpackHalf2x16 = 62,
    GlslLength = 66,
    GlslDistance = 67,
    GlslCross = 68,
    GlslNormalize = 69,
    GlslFaceForward = 70,
    GlslReflect = 71,
    GlslRefract = 72,
    GlslFindILsb = 73,
    GlslFindSMsb = 74,
    GlslFindUMsb = 75,
};
}  // namespace spv

// ---- Arena and growable arrays ------------------------------------------------

class Arena
{
public:
    explicit Arena(cyber::memory::Allocator& allocator) noexcept : m_allocator(&allocator) {}
    ~Arena();
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    [[nodiscard]] void* Allocate(usize bytes, usize alignment) noexcept;
    [[nodiscard]] bool Failed() const noexcept
    {
        return m_failed;
    }

private:
    struct Chunk
    {
        Chunk* next;
        usize used;
        usize capacity;
    };
    cyber::memory::Allocator* m_allocator;
    Chunk* m_head = nullptr;
    bool m_failed = false;
};

// A growable array of trivially copyable values in an arena. Out-of-range reads
// return a zeroed element instead of reading past the end, so a malformed input
// that slips past a check degrades into a diagnostic rather than a crash.
template <typename T>
class Vec
{
public:
    void Init(Arena* arena) noexcept
    {
        m_arena = arena;
    }

    bool Push(const T& value) noexcept
    {
        if (m_size == m_capacity && !Grow(m_size + 1))
        {
            return false;
        }
        m_data[m_size++] = value;
        return true;
    }
    bool Resize(u32 size) noexcept
    {
        if (size > m_capacity && !Grow(size))
        {
            return false;
        }
        for (u32 index = m_size; index < size; ++index)
        {
            m_data[index] = T{};
        }
        m_size = size;
        return true;
    }
    void Shrink(u32 size) noexcept
    {
        if (size < m_size)
        {
            m_size = size;
        }
    }
    [[nodiscard]] T& operator[](u32 index) noexcept
    {
        if (index >= m_size)
        {
            m_dummy = T{};
            return m_dummy;
        }
        return m_data[index];
    }
    [[nodiscard]] const T& operator[](u32 index) const noexcept
    {
        return index < m_size ? m_data[index] : m_dummy;
    }
    [[nodiscard]] u32 Size() const noexcept
    {
        return m_size;
    }
    [[nodiscard]] T* Data() noexcept
    {
        return m_data;
    }
    [[nodiscard]] const T* Data() const noexcept
    {
        return m_data;
    }
    [[nodiscard]] T& Back() noexcept
    {
        return (*this)[m_size == 0 ? 0 : m_size - 1];
    }
    void Clear() noexcept
    {
        m_size = 0;
    }

private:
    bool Grow(u32 minimum) noexcept
    {
        if (m_arena == nullptr)
        {
            return false;
        }
        u32 capacity = m_capacity < 16 ? 16 : m_capacity * 2;
        while (capacity < minimum)
        {
            capacity *= 2;
        }
        T* data = static_cast<T*>(m_arena->Allocate(sizeof(T) * capacity, alignof(T)));
        if (data == nullptr)
        {
            return false;
        }
        if (m_size > 0)
        {
            std::memcpy(static_cast<void*>(data), m_data, sizeof(T) * m_size);
        }
        m_data = data;
        m_capacity = capacity;
        return true;
    }

    Arena* m_arena = nullptr;
    T* m_data = nullptr;
    u32 m_size = 0;
    u32 m_capacity = 0;
    mutable T m_dummy{};
};

// ---- Tokens ---------------------------------------------------------------------

enum class Tok : u8
{
    End = 0,
    Ident,
    Int,
    Float,
    Punct,
};

struct Token
{
    Tok kind = Tok::End;
    bool isUnsigned = false;
    const char* text = nullptr;
    u32 length = 0;
    u32 line = 1;
    u32 column = 1;
    u64 intValue = 0;
    double floatValue = 0.0;
};

struct Name
{
    const char* text = nullptr;
    u32 length = 0;

    [[nodiscard]] bool Is(const char* literal) const noexcept
    {
        const usize n = std::strlen(literal);
        return n == length && (n == 0 || std::memcmp(text, literal, n) == 0);
    }
    [[nodiscard]] bool Equals(const Name& other) const noexcept
    {
        return length == other.length &&
               (length == 0 || std::memcmp(text, other.text, length) == 0);
    }
};

// ---- Types ------------------------------------------------------------------------

// Index into the type table. 0 is "no type".
using TypeId = u32;

enum class TK : u8
{
    None = 0,
    Void,
    Bool,
    Int,
    Uint,
    Float,
    Vector,   // a = element, b = count
    Matrix,   // a = column (vector) type, b = column count; HLSL rows = SPIR-V columns
    Array,    // a = element, b = length, c = stride (0 = logical)
    RtArray,  // a = element, c = stride
    Struct,   // a = struct definition, b = layout, c = block flag
    Image,    // a = sampled type, b = dim, c = arrayed | ms << 1 | sampled << 2 | depth << 4
    Sampler,
    SampledImage,  // a = image
    Pointer,       // a = storage class, b = pointee
    Function,      // a = return, b = first parameter in m_funcTypeParams, c = count
};

enum Layout : u8
{
    LayoutLogical = 0,
    LayoutCBuffer = 1,
    LayoutStructured = 2,
};

struct TypeEntry
{
    TK kind = TK::None;
    u32 a = 0;
    u32 b = 0;
    u32 c = 0;
    u32 id = 0;
};

// Interpolation qualifiers on a varying.
enum Interp : u8
{
    InterpDefault = 0,
    InterpFlat = 1,
    InterpNoPerspective = 2,
    InterpCentroid = 4,
    InterpSample = 8,
};

struct Semantic
{
    Name name;  // without the index
    u32 index = 0;
    bool present = false;
};

struct StructMember
{
    Name name;
    TypeId type = 0;  // logical
    Semantic semantic;
    u8 interp = 0;
    bool hasPackOffset = false;
    u32 packOffsetBytes = 0;
};

struct StructDef
{
    Name name;
    u32 firstMember = 0;
    u32 memberCount = 0;
    // Layout-specific SPIR-V types, created on demand.
    TypeId laidOut[3][2] = {};
};

// ---- Resources ----------------------------------------------------------------

enum class Res : u8
{
    None = 0,
    Texture1D,
    Texture1DArray,
    Texture2D,
    Texture2DArray,
    Texture2DMS,
    Texture3D,
    TextureCube,
    TextureCubeArray,
    RWTexture1D,
    RWTexture1DArray,
    RWTexture2D,
    RWTexture2DArray,
    RWTexture3D,
    StructuredBuffer,
    RWStructuredBuffer,
    ByteAddressBuffer,
    RWByteAddressBuffer,
    SamplerState,
    SamplerComparisonState,
    CBuffer,  // a `cbuffer` block or `ConstantBuffer<T>`
};

inline constexpr u32 kUnboundedArray = 0xFFFFFFFFU;

struct ResourceDecl
{
    Name name;
    Res kind = Res::None;
    TypeId element = 0;  // logical element type; for CBuffer, the block's struct type
    u32 structDef = 0;   // CBuffer: definition index + 1
    bool cbufferMembersGlobal = false;  // `cbuffer X { ... }` rather than ConstantBuffer<T>
    RegisterClass registerClass = RegisterClass::ShaderResource;
    u32 registerIndex = 0;
    u32 space = 0;
    bool hasRegister = false;
    u32 arraySize = 0;  // 0 = not an array
    // Resolved on first use.
    bool resolved = false;
    ResolvedBinding binding{};
    u32 variable = 0;         // SPIR-V id of the descriptor variable
    TypeId variableType = 0;  // pointee type of the variable
    TypeId blockType = 0;     // buffers: the block struct
    u32 storage = 0;
    bool binds = false;  // the variable is an array of descriptors
};

// A descriptor variable shared by every HLSL declaration that lands in one binding.
struct BindingVar
{
    u32 set = 0;
    u32 binding = 0;
    u32 variable = 0;
    TypeId pointee = 0;
};

// ---- Values -------------------------------------------------------------------

enum class VK : u8
{
    None = 0,
    R,         // an SSA value
    Ptr,       // a pointer: `id` is the pointer, `type` the pointee
    Swz,       // a swizzle of a vector pointer: `id` pointer, `type` the vector
    Texel,     // a storage image texel: `id` image pointer, `aux` coordinate value
    Resource,  // a resource declaration, optionally indexed
};

struct Val
{
    VK kind = VK::None;
    u32 id = 0;
    TypeId type = 0;
    u32 storage = 0;
    u8 swizzle[4] = {};
    u8 swizzleCount = 0;
    bool readOnly = false;
    bool nonUniform = false;
    // Scalar constant folding.
    bool isConst = false;
    u32 constBits = 0;
    // Resource values.
    u32 resource = 0;  // declaration index + 1
    u32 aux = 0;       // Resource: index value id (0 = none); Texel: coordinate id
};

// ---- Symbols and functions --------------------------------------------------------

enum class SK : u8
{
    Var,
    Const,
    CBufferMember,
    Resource,
    Struct,
    Function,
};

struct Symbol
{
    Name name;
    SK kind = SK::Var;
    Val value{};
    u32 index = 0;   // resource / struct / first function decl
    u32 member = 0;  // CBufferMember: member index
    bool precise = false;
};

enum class ParamDir : u8
{
    In,
    Out,
    InOut,
};

struct Param
{
    Name name;
    TypeId type = 0;
    ParamDir dir = ParamDir::In;
    Semantic semantic;
    u8 interp = 0;
};

struct FuncDecl
{
    Name name;
    TypeId returnType = 0;
    Semantic returnSemantic;
    u32 firstParam = 0;
    u32 paramCount = 0;
    u32 bodyToken = 0;  // index of '{', 0 for a prototype
    u32 spirvId = 0;
    TypeId functionType = 0;
    bool compiling = false;
    u32 numThreads[3] = {0, 0, 0};
    u32 declToken = 0;
};

struct LoopTargets
{
    u32 merge = 0;
    u32 continueTarget = 0;
};

// Per-function emission state.
struct FuncState
{
    Vec<u32> variables;
    Vec<u32> body;
    bool blockOpen = false;
    TypeId returnType = 0;
    Vec<LoopTargets> loops;
};

class Compiler
{
public:
    Compiler(const CompileDesc& desc, cyber::memory::Allocator& allocator,
             CompileOutput* out) noexcept;

    [[nodiscard]] bool Run() noexcept;
    [[nodiscard]] bool OutOfMemory() const noexcept
    {
        return m_arena.Failed();
    }

    // ---- hlsl_spirv_core.cpp: lexing, module, types, constants, conversions ----
    bool Lex() noexcept;
    bool Fail(const Token& at, const char* message) noexcept;
    bool FailHere(const char* message) noexcept;
    [[nodiscard]] bool HasError() const noexcept
    {
        return m_error;
    }

    const Token& Peek(u32 ahead = 0) const noexcept;
    const Token& Next() noexcept;
    bool IsPunct(const char* text, u32 ahead = 0) const noexcept;
    bool IsIdent(const char* text, u32 ahead = 0) const noexcept;
    bool Accept(const char* punct) noexcept;
    bool Expect(const char* punct, const char* message) noexcept;
    Name ExpectIdent(const char* message) noexcept;
    bool SkipBalanced(const char* open, const char* close) noexcept;

    u32 NewId() noexcept
    {
        return m_nextId++;
    }
    void Inst(Vec<u32>& stream, u32 op, const u32* operands, u32 count) noexcept;
    void Inst(Vec<u32>& stream, u32 op, std::initializer_list<u32> operands) noexcept;
    void InstString(Vec<u32>& stream, u32 op, std::initializer_list<u32> head, const char* text,
                    u32 length, const u32* tail, u32 tailCount) noexcept;
    void Capability(u32 capability) noexcept;
    void Decorate(u32 target, u32 decoration, std::initializer_list<u32> extra = {}) noexcept;
    void MemberDecorate(u32 type, u32 member, u32 decoration,
                        std::initializer_list<u32> extra = {}) noexcept;

    TypeId Intern(TK kind, u32 a = 0, u32 b = 0, u32 c = 0) noexcept;
    TypeId VoidType() noexcept
    {
        return Intern(TK::Void);
    }
    TypeId BoolType() noexcept
    {
        return Intern(TK::Bool);
    }
    TypeId IntType() noexcept
    {
        return Intern(TK::Int);
    }
    TypeId UintType() noexcept
    {
        return Intern(TK::Uint);
    }
    TypeId FloatType() noexcept
    {
        return Intern(TK::Float);
    }
    TypeId VectorOf(TypeId scalar, u32 count) noexcept;
    TypeId MatrixOf(TypeId scalar, u32 rows, u32 columns) noexcept;
    TypeId PointerTo(u32 storage, TypeId pointee) noexcept;
    TypeId ArrayOf(TypeId element, u32 length, u32 stride) noexcept;
    TypeId RuntimeArrayOf(TypeId element, u32 stride) noexcept;
    TypeId FunctionType(TypeId result, const TypeId* params, u32 count) noexcept;
    u32 SpirvType(TypeId type) noexcept;
    [[nodiscard]] const TypeEntry& T(TypeId type) const noexcept
    {
        return m_types[type];
    }

    [[nodiscard]] bool IsScalar(TypeId type) const noexcept;
    [[nodiscard]] bool IsNumericScalarOrVector(TypeId type) const noexcept;
    [[nodiscard]] TypeId ScalarOf(TypeId type) const noexcept;
    [[nodiscard]] u32 ComponentCount(TypeId type) const noexcept;  // 1 for scalars
    [[nodiscard]] TypeId WithScalar(TypeId shape, TypeId scalar) noexcept;
    [[nodiscard]] TypeId WithCount(TypeId scalar, u32 count) noexcept;  // 1 -> scalar
    [[nodiscard]] bool IsFloatish(TypeId type) const noexcept;
    [[nodiscard]] bool IsSigned(TypeId type) const noexcept;
    [[nodiscard]] bool IsBoolish(TypeId type) const noexcept;
    [[nodiscard]] bool IsInteger(TypeId type) const noexcept;
    [[nodiscard]] TypeId LogicalOf(TypeId type) noexcept;
    TypeId LaidOut(TypeId logical, Layout layout, bool block) noexcept;
    bool SizeAndAlign(TypeId logical, Layout layout, u32* size, u32* align) noexcept;
    TypeId StructType(u32 def, Layout layout, bool block) noexcept;

    u32 ConstantBits(TypeId scalar, u32 bits) noexcept;
    u32 ConstUint(u32 value) noexcept
    {
        return ConstantBits(UintType(), value);
    }
    u32 ConstInt(int value) noexcept
    {
        return ConstantBits(IntType(), static_cast<u32>(value));
    }
    u32 ConstFloat(float value) noexcept;
    u32 ConstantComposite(TypeId type, const u32* ids, u32 count) noexcept;
    u32 ConstantNull(TypeId type) noexcept;
    Val ConstVal(TypeId scalar, u32 bits) noexcept;
    Val Splat(Val scalar, TypeId vectorType) noexcept;

    // Code emission into the current function.
    Vec<u32>& Body() noexcept;
    void Open() noexcept;
    u32 Op(u32 op, TypeId result, std::initializer_list<u32> operands) noexcept;
    u32 OpN(u32 op, TypeId result, const u32* operands, u32 count) noexcept;
    void OpVoid(u32 op, std::initializer_list<u32> operands) noexcept;
    u32 Ext(u32 instruction, TypeId result, std::initializer_list<u32> operands) noexcept;
    void Label(u32 id) noexcept;
    void Branch(u32 target) noexcept;
    void Terminate(u32 op, std::initializer_list<u32> operands = {}) noexcept;
    u32 NewLocal(TypeId type) noexcept;

    Val Load(const Val& value) noexcept;
    Val Rvalue(const Val& value) noexcept;
    bool Store(const Val& target, Val value) noexcept;
    Val Convert(Val value, TypeId target) noexcept;
    Val ConvertLayout(Val value, TypeId target) noexcept;
    Val ToBool(Val value) noexcept;

    // ---- hlsl_spirv_decl.cpp: declarations, statements, functions, entry ----
    bool ParseTopLevel() noexcept;
    TypeId ParseType(bool allowVoid) noexcept;
    TypeId TypeFromName(const Name& name) noexcept;
    bool IsTypeStart(u32 ahead = 0) noexcept;
    bool ParseSemantic(Semantic* out) noexcept;
    bool ParseRegister(ResourceDecl* decl) noexcept;
    bool ParseStructBody(u32 defIndex) noexcept;
    bool ParseResourceDecl(Res kind) noexcept;
    bool ParseCBuffer() noexcept;
    bool ParseFunction(TypeId returnType, Name name, const u32* numThreads, u32 declToken) noexcept;
    bool ParseGlobalVariable(TypeId type, Name name, bool isStatic, bool isConst,
                             bool isGroupShared) noexcept;
    bool ParseArraySuffix(TypeId* type) noexcept;
    bool ConstantU32(u32* out) noexcept;
    bool CompileFunction(u32 decl) noexcept;
    bool ParseBlock() noexcept;
    bool ParseStatement() noexcept;
    bool ParseStatementInner() noexcept;
    bool ParseLocalDeclaration() noexcept;
    bool ParseIf() noexcept;
    bool ParseFor(u32 loopControl) noexcept;
    bool ParseWhile(u32 loopControl) noexcept;
    bool ParseDoWhile(u32 loopControl) noexcept;
    bool ParseReturn() noexcept;
    bool EmitEntryPoint(u32 decl) noexcept;
    void PushScope() noexcept
    {
        (void)m_scopes.Push(m_symbols.Size());
    }
    void PopScope() noexcept;
    const Symbol* Lookup(const Name& name) const noexcept;
    bool Declare(const Symbol& symbol) noexcept;

    // ---- hlsl_spirv_expr.cpp: expressions, intrinsics, resources ----
    Val ParseExpression() noexcept;
    Val ParseAssignment() noexcept;
    Val ParseTernary() noexcept;
    Val ParseBinary(int precedence) noexcept;
    Val ParseUnary() noexcept;
    Val ParseUnaryInner() noexcept;
    Val ParsePostfix(Val value) noexcept;
    Val ParsePrimary() noexcept;
    Val ParseInitializer(TypeId type) noexcept;
    bool ParseArguments(Val* args, u32 capacity, u32* count) noexcept;
    Val Binary(const char* op, Val lhs, Val rhs, const Token& at) noexcept;
    Val Arithmetic(const char* op, Val lhs, Val rhs, const Token& at) noexcept;
    Val Construct(TypeId type, const Val* args, u32 count, const Token& at) noexcept;
    Val CallFunction(const Name& name, Val* args, u32 count, const Token& at) noexcept;
    Val CallIntrinsic(const Name& name, Val* args, u32 count, const Token& at,
                      bool* handled) noexcept;
    Val CallMethod(const Val& object, const Name& method, Val* args, u32 count,
                   const Token& at) noexcept;
    Val Member(const Val& object, const Name& member, const Token& at) noexcept;
    Val Index(const Val& object, Val index, const Token& at) noexcept;
    Val Swizzle(const Val& value, const Name& name, const Token& at) noexcept;
    Val ResourceElement(const Val& resource, const Token& at) noexcept;
    bool BindResource(u32 decl, const Token& at) noexcept;
    Val CBufferMember(u32 decl, u32 member, const Token& at) noexcept;
    TypeId CommonType(const Val* values, u32 count, bool vectorMin) noexcept;
    Val Atomic(const char* name, const Val& dest, Val* args, u32 count, const Token& at) noexcept;
    Val AtomicOnPointer(u32 opcode, const Val& pointer, Val value, Val comparator,
                        const Token& at) noexcept;
    bool FlattenScalars(const Val& value, Vec<Val>& out) noexcept;
    Val BuildFromScalars(TypeId type, const Vec<Val>& scalars, u32* cursor,
                         const Token& at) noexcept;

    // ---- state ----
    const CompileDesc& m_desc;
    CompileOutput* m_out;
    Arena m_arena;
    bool m_error = false;

    Vec<Token> m_tokens;
    u32 m_cursor = 0;

    u32 m_nextId = 1;
    Vec<u32> m_capabilities;
    Vec<u32> m_entryPoints;
    Vec<u32> m_executionModes;
    Vec<u32> m_debug;
    Vec<u32> m_annotations;
    Vec<u32> m_globals;
    Vec<u32> m_functions;
    Vec<u32> m_interface;
    u32 m_glsl = 0;

    Vec<TypeEntry> m_types;
    Vec<TypeId> m_funcTypeParams;
    struct ConstKey
    {
        u32 op;
        TypeId type;
        u32 first;  // into m_constOperands
        u32 count;
        u32 id;
    };
    Vec<ConstKey> m_constants;
    Vec<u32> m_constOperands;

    Vec<StructDef> m_structs;
    Vec<StructMember> m_members;
    Vec<ResourceDecl> m_resources;
    Vec<BindingVar> m_bindingVars;
    Vec<FuncDecl> m_funcs;
    Vec<Param> m_params;
    Vec<Symbol> m_symbols;
    Vec<u32> m_scopes;
    u32 m_globalSymbolCount = 0;

    Vec<FuncState*> m_funcStack;
    // A scratch function so global initialisers can be folded through the normal
    // expression path; it must fold to constants, never emit.
    FuncState m_globalScope;
    bool m_atGlobalScope = true;
    bool m_precise = false;
    // Expression and statement nesting, bounded so hostile input cannot exhaust the stack.
    u32 m_depth = 0;
};

// Float bit helpers.
inline u32 FloatBits(float value) noexcept
{
    u32 bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
inline float BitsFloat(u32 bits) noexcept
{
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace cyber::rhi::detail::hlsl
