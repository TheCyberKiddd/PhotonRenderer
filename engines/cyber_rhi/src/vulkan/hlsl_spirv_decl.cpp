// Declarations, statements, functions and the entry-point wrapper for the HLSL
// compiler (hlsl_spirv_internal.h).

#include "hlsl_spirv_internal.h"

#include <cstdio>
#include <new>

namespace cyber::rhi::detail::hlsl
{

namespace
{

char Upper(char c) noexcept
{
    return c >= 'a' && c <= 'z' ? static_cast<char>(c - 'a' + 'A') : c;
}

// Case-insensitive comparison of a semantic name (without index) to `literal`.
bool SemanticIs(const Semantic& semantic, const char* literal) noexcept
{
    const usize length = std::strlen(literal);
    if (!semantic.present || semantic.name.length != length)
    {
        return false;
    }
    for (usize index = 0; index < length; ++index)
    {
        if (Upper(semantic.name.text[index]) != literal[index])
        {
            return false;
        }
    }
    return true;
}

struct ResourceName
{
    const char* name;
    Res kind;
};

constexpr ResourceName kResourceNames[] = {
    {"Texture1D", Res::Texture1D},
    {"Texture1DArray", Res::Texture1DArray},
    {"Texture2D", Res::Texture2D},
    {"Texture2DArray", Res::Texture2DArray},
    {"Texture2DMS", Res::Texture2DMS},
    {"Texture3D", Res::Texture3D},
    {"TextureCube", Res::TextureCube},
    {"TextureCubeArray", Res::TextureCubeArray},
    {"RWTexture1D", Res::RWTexture1D},
    {"RWTexture1DArray", Res::RWTexture1DArray},
    {"RWTexture2D", Res::RWTexture2D},
    {"RWTexture2DArray", Res::RWTexture2DArray},
    {"RWTexture3D", Res::RWTexture3D},
    {"StructuredBuffer", Res::StructuredBuffer},
    {"RWStructuredBuffer", Res::RWStructuredBuffer},
    {"ByteAddressBuffer", Res::ByteAddressBuffer},
    {"RWByteAddressBuffer", Res::RWByteAddressBuffer},
    {"SamplerState", Res::SamplerState},
    {"SamplerComparisonState", Res::SamplerComparisonState},
    {"ConstantBuffer", Res::CBuffer},
};

Res ResourceKindOf(const Token& token) noexcept
{
    if (token.kind != Tok::Ident)
    {
        return Res::None;
    }
    for (const ResourceName& entry : kResourceNames)
    {
        const usize length = std::strlen(entry.name);
        if (token.length == length && std::memcmp(token.text, entry.name, length) == 0)
        {
            return entry.kind;
        }
    }
    return Res::None;
}

RegisterClass ClassOf(Res kind) noexcept
{
    switch (kind)
    {
    case Res::CBuffer:
        return RegisterClass::ConstantBuffer;
    case Res::RWTexture1D:
    case Res::RWTexture1DArray:
    case Res::RWTexture2D:
    case Res::RWTexture2DArray:
    case Res::RWTexture3D:
    case Res::RWStructuredBuffer:
    case Res::RWByteAddressBuffer:
        return RegisterClass::UnorderedAccess;
    case Res::SamplerState:
    case Res::SamplerComparisonState:
        return RegisterClass::Sampler;
    default:
        return RegisterClass::ShaderResource;
    }
}

constexpr const char* kQualifiers[] = {
    "const",     "static",        "precise",  "groupshared", "uniform", "extern",
    "volatile",  "inline",        "in",       "out",         "inout",   "nointerpolation",
    "linear",    "noperspective", "centroid", "sample",      "snorm",   "unorm",
    "row_major", "column_major",
};

}  // namespace

// ---- Symbols ------------------------------------------------------------------

void Compiler::PopScope() noexcept
{
    if (m_scopes.Size() == 0)
    {
        return;
    }
    m_symbols.Shrink(m_scopes.Back());
    m_scopes.Shrink(m_scopes.Size() - 1);
}

const Symbol* Compiler::Lookup(const Name& name) const noexcept
{
    for (u32 index = m_symbols.Size(); index > 0; --index)
    {
        const Symbol& symbol = m_symbols[index - 1];
        if (symbol.name.Equals(name))
        {
            return &symbol;
        }
    }
    return nullptr;
}

bool Compiler::Declare(const Symbol& symbol) noexcept
{
    const u32 scopeStart = m_scopes.Size() == 0 ? 0 : m_scopes.Back();
    for (u32 index = scopeStart; index < m_symbols.Size(); ++index)
    {
        if (m_symbols[index].name.Equals(symbol.name))
        {
            return FailHere("redefinition of a name in the same scope");
        }
    }
    return m_symbols.Push(symbol) || FailHere("out of memory");
}

// ---- Types --------------------------------------------------------------------

TypeId Compiler::TypeFromName(const Name& name) noexcept
{
    if (name.Is("void"))
    {
        return VoidType();
    }
    struct Base
    {
        const char* text;
        TK kind;
    };
    constexpr Base kBases[] = {
        {"min16float", TK::Float}, {"min10float", TK::Float}, {"min16uint", TK::Uint},
        {"min16int", TK::Int},     {"min12int", TK::Int},     {"dword", TK::Uint},
        {"float", TK::Float},      {"half", TK::Float},       {"uint", TK::Uint},
        {"bool", TK::Bool},        {"int", TK::Int},
    };
    for (const Base& base : kBases)
    {
        const u32 length = static_cast<u32>(std::strlen(base.text));
        if (name.length < length || std::memcmp(name.text, base.text, length) != 0)
        {
            continue;
        }
        const char* rest = name.text + length;
        const u32 restLength = name.length - length;
        const TypeId scalar = Intern(base.kind);
        if (restLength == 0)
        {
            return scalar;
        }
        if (restLength == 1 && rest[0] >= '1' && rest[0] <= '4')
        {
            return VectorOf(scalar, static_cast<u32>(rest[0] - '0'));
        }
        if (restLength == 3 && rest[0] >= '1' && rest[0] <= '4' && rest[1] == 'x' &&
            rest[2] >= '1' && rest[2] <= '4')
        {
            const u32 rows = static_cast<u32>(rest[0] - '0');
            const u32 columns = static_cast<u32>(rest[2] - '0');
            if (base.kind != TK::Float)
            {
                (void)FailHere("only float matrices are supported by the Vulkan HLSL compiler");
                return 0;
            }
            if (columns == 1)
            {
                // floatRx1 is a column; treat it as a vector of R.
                return VectorOf(scalar, rows);
            }
            if (rows == 1)
            {
                return VectorOf(scalar, columns);
            }
            return MatrixOf(scalar, rows, columns);
        }
        return 0;
    }
    const Symbol* symbol = Lookup(name);
    if (symbol != nullptr && symbol->kind == SK::Struct)
    {
        return StructType(symbol->index, LayoutLogical, false);
    }
    return 0;
}

bool Compiler::IsTypeStart(u32 ahead) noexcept
{
    const Token& token = Peek(ahead);
    if (token.kind != Tok::Ident)
    {
        return false;
    }
    const Name name{token.text, token.length};
    if (name.Is("vector") || name.Is("matrix"))
    {
        return true;
    }
    const bool hadError = m_error;
    const TypeId type = TypeFromName(name);
    if (!hadError && m_error)
    {
        return true;  // a type name with an error; let the caller report it
    }
    return type != 0;
}

TypeId Compiler::ParseType(bool allowVoid) noexcept
{
    if (m_error)
    {
        return 0;
    }
    const Token& token = Peek();
    if (token.kind != Tok::Ident)
    {
        (void)FailHere("expected a type");
        return 0;
    }
    const Name name{token.text, token.length};
    ++m_cursor;
    if (name.Is("vector") || name.Is("matrix"))
    {
        const bool isMatrix = name.Is("matrix");
        if (!Accept("<"))
        {
            // Bare `vector`/`matrix` mean float4 and float4x4.
            return isMatrix ? MatrixOf(FloatType(), 4, 4) : VectorOf(FloatType(), 4);
        }
        const TypeId scalar = ParseType(false);
        if (scalar == 0 || !IsScalar(scalar))
        {
            (void)FailHere("expected a scalar type in a vector or matrix template");
            return 0;
        }
        u32 first = 0;
        u32 second = 0;
        if (!Expect(",", "expected ','") || !ConstantU32(&first))
        {
            return 0;
        }
        if (isMatrix && (!Expect(",", "expected ','") || !ConstantU32(&second)))
        {
            return 0;
        }
        if (!Expect(">", "expected '>'"))
        {
            return 0;
        }
        if (first < 1 || first > 4 || (isMatrix && (second < 1 || second > 4)))
        {
            (void)FailHere("vector and matrix dimensions are 1 to 4");
            return 0;
        }
        if (!isMatrix)
        {
            return VectorOf(scalar, first);
        }
        if (T(scalar).kind != TK::Float)
        {
            (void)FailHere("only float matrices are supported by the Vulkan HLSL compiler");
            return 0;
        }
        if (first == 1 || second == 1)
        {
            return VectorOf(scalar, first == 1 ? second : first);
        }
        return MatrixOf(scalar, first, second);
    }
    if (name.Is("double") || name.Is("int64_t") || name.Is("uint64_t") || name.Is("float16_t") ||
        name.Is("int16_t") || name.Is("uint16_t") || name.Is("float64_t"))
    {
        (void)Fail(token, "64-bit and 16-bit types are not supported by the Vulkan HLSL compiler");
        return 0;
    }
    const TypeId type = TypeFromName(name);
    if (m_error)
    {
        return 0;
    }
    if (type == 0)
    {
        (void)Fail(token, "unknown type name");
        return 0;
    }
    if (!allowVoid && T(type).kind == TK::Void)
    {
        (void)Fail(token, "void is not valid here");
        return 0;
    }
    return type;
}

bool Compiler::ConstantU32(u32* out) noexcept
{
    const bool wasGlobal = m_atGlobalScope;
    m_atGlobalScope = true;
    // Shift level and tighter: `vector<float, 3>` must not read `3 > ...`.
    const Val value = ParseBinary(8);
    m_atGlobalScope = wasGlobal;
    if (m_error)
    {
        return false;
    }
    if (value.kind != VK::R || !value.isConst || !IsScalar(value.type) || IsFloatish(value.type))
    {
        return FailHere("expected a constant integer expression");
    }
    if (IsSigned(value.type) && static_cast<int>(value.constBits) < 0)
    {
        return FailHere("expected a non-negative constant");
    }
    *out = value.constBits;
    return true;
}

bool Compiler::ParseArraySuffix(TypeId* type) noexcept
{
    u32 dims[4] = {};
    u32 count = 0;
    while (IsPunct("["))
    {
        ++m_cursor;
        if (count == 4)
        {
            return FailHere("arrays are limited to four dimensions");
        }
        u32 length = 0;
        if (!ConstantU32(&length) || !Expect("]", "expected ']'"))
        {
            return false;
        }
        if (length == 0)
        {
            return FailHere("array length must be positive");
        }
        dims[count++] = length;
    }
    for (u32 index = count; index > 0; --index)
    {
        *type = ArrayOf(*type, dims[index - 1], 0);
    }
    return true;
}

bool Compiler::ParseSemantic(Semantic* out) noexcept
{
    const Name name = ExpectIdent("expected a semantic");
    if (m_error)
    {
        return false;
    }
    u32 digits = 0;
    while (digits < name.length && name.text[name.length - 1 - digits] >= '0' &&
           name.text[name.length - 1 - digits] <= '9')
    {
        ++digits;
    }
    out->present = true;
    out->name = Name{name.text, name.length - digits};
    out->index = 0;
    for (u32 index = name.length - digits; index < name.length; ++index)
    {
        out->index = out->index * 10 + static_cast<u32>(name.text[index] - '0');
    }
    return true;
}

bool Compiler::ParseRegister(ResourceDecl* decl) noexcept
{
    // After `register`.
    if (!Expect("(", "expected '(' after register"))
    {
        return false;
    }
    const Token& token = Peek();
    const Name reg = ExpectIdent("expected a register such as t0");
    if (m_error)
    {
        return false;
    }
    if (reg.length < 2)
    {
        return Fail(token, "malformed register");
    }
    const char letter = static_cast<char>(reg.text[0] | 0x20);
    RegisterClass registerClass = RegisterClass::ShaderResource;
    switch (letter)
    {
    case 'b':
        registerClass = RegisterClass::ConstantBuffer;
        break;
    case 't':
        registerClass = RegisterClass::ShaderResource;
        break;
    case 'u':
        registerClass = RegisterClass::UnorderedAccess;
        break;
    case 's':
        registerClass = RegisterClass::Sampler;
        break;
    default:
        return Fail(token, "unknown register class");
    }
    u32 index = 0;
    for (u32 c = 1; c < reg.length; ++c)
    {
        if (reg.text[c] < '0' || reg.text[c] > '9' || index > 100000000U)
        {
            return Fail(token, "malformed register");
        }
        index = index * 10 + static_cast<u32>(reg.text[c] - '0');
    }
    if (registerClass != ClassOf(decl->kind))
    {
        return Fail(token, "resource uses the wrong register class");
    }
    decl->registerClass = registerClass;
    decl->registerIndex = index;
    decl->space = 0;
    decl->hasRegister = true;
    if (Accept(","))
    {
        const Token& spaceToken = Peek();
        const Name space = ExpectIdent("expected space<n>");
        if (m_error)
        {
            return false;
        }
        if (space.length < 6 || std::memcmp(space.text, "space", 5) != 0)
        {
            return Fail(spaceToken, "expected space<n>");
        }
        u32 value = 0;
        for (u32 c = 5; c < space.length; ++c)
        {
            if (space.text[c] < '0' || space.text[c] > '9' || value > 1000000U)
            {
                return Fail(spaceToken, "malformed register space");
            }
            value = value * 10 + static_cast<u32>(space.text[c] - '0');
        }
        decl->space = value;
    }
    return Expect(")", "expected ')' after register");
}

// ---- Top level ------------------------------------------------------------------

bool Compiler::ParseStructBody(u32 defIndex) noexcept
{
    if (!Expect("{", "expected '{'"))
    {
        return false;
    }
    const u32 first = m_members.Size();
    while (!m_error && !IsPunct("}"))
    {
        if (Peek().kind == Tok::End)
        {
            return FailHere("unterminated structure");
        }
        u8 interp = 0;
        while (Peek().kind == Tok::Ident)
        {
            if (IsIdent("nointerpolation"))
            {
                interp |= InterpFlat;
            }
            else if (IsIdent("noperspective"))
            {
                interp |= InterpNoPerspective;
            }
            else if (IsIdent("centroid"))
            {
                interp |= InterpCentroid;
            }
            else if (IsIdent("sample"))
            {
                interp |= InterpSample;
            }
            else if (IsIdent("linear") || IsIdent("precise"))
            {
            }
            else if (IsIdent("row_major"))
            {
                return FailHere("row_major matrices are not supported by the Vulkan HLSL compiler");
            }
            else if (IsIdent("column_major"))
            {
            }
            else
            {
                break;
            }
            ++m_cursor;
        }
        const TypeId base = ParseType(false);
        if (base == 0)
        {
            return false;
        }
        for (;;)
        {
            StructMember member{};
            member.name = ExpectIdent("expected a member name");
            member.type = base;
            member.interp = interp;
            if (!ParseArraySuffix(&member.type))
            {
                return false;
            }
            if (Accept(":"))
            {
                if (IsIdent("packoffset"))
                {
                    ++m_cursor;
                    if (!Expect("(", "expected '('"))
                    {
                        return false;
                    }
                    const Token& token = Peek();
                    const Name reg = ExpectIdent("expected c<n>");
                    if (m_error || reg.length < 2 || (reg.text[0] | 0x20) != 'c')
                    {
                        return Fail(token, "malformed packoffset");
                    }
                    u32 value = 0;
                    for (u32 c = 1; c < reg.length; ++c)
                    {
                        if (reg.text[c] < '0' || reg.text[c] > '9' || value > 1000000U)
                        {
                            return Fail(token, "malformed packoffset");
                        }
                        value = value * 10 + static_cast<u32>(reg.text[c] - '0');
                    }
                    u32 component = 0;
                    if (Accept("."))
                    {
                        const Name comp = ExpectIdent("expected a component");
                        if (m_error || comp.length != 1)
                        {
                            return FailHere("malformed packoffset component");
                        }
                        const char c = comp.text[0];
                        component = c == 'x'   ? 0U
                                    : c == 'y' ? 1U
                                    : c == 'z' ? 2U
                                    : c == 'w' ? 3U
                                               : 4U;
                        if (component == 4)
                        {
                            return FailHere("malformed packoffset component");
                        }
                    }
                    member.hasPackOffset = true;
                    member.packOffsetBytes = value * 16 + component * 4;
                    if (!Expect(")", "expected ')'"))
                    {
                        return false;
                    }
                }
                else if (!ParseSemantic(&member.semantic))
                {
                    return false;
                }
            }
            if (!m_members.Push(member))
            {
                return FailHere("out of memory");
            }
            if (!Accept(","))
            {
                break;
            }
        }
        if (!Expect(";", "expected ';' after a member"))
        {
            return false;
        }
    }
    if (!Expect("}", "expected '}'"))
    {
        return false;
    }
    StructDef& def = m_structs[defIndex];
    def.firstMember = first;
    def.memberCount = m_members.Size() - first;
    if (def.memberCount == 0)
    {
        return FailHere("empty structures are not supported");
    }
    if (def.memberCount > 64)
    {
        return FailHere("structures are limited to 64 members");
    }
    return true;
}

bool Compiler::ParseResourceDecl(Res kind) noexcept
{
    const Token& start = Next();  // the type keyword
    ResourceDecl decl{};
    decl.kind = kind;
    decl.registerClass = ClassOf(kind);
    const bool isTexture = kind != Res::StructuredBuffer && kind != Res::RWStructuredBuffer &&
                           kind != Res::ByteAddressBuffer && kind != Res::RWByteAddressBuffer &&
                           kind != Res::SamplerState && kind != Res::SamplerComparisonState &&
                           kind != Res::CBuffer;
    if (isTexture)
    {
        decl.element = VectorOf(FloatType(), 4);
    }
    if (Accept("<"))
    {
        while (IsIdent("unorm") || IsIdent("snorm"))
        {
            ++m_cursor;
        }
        const TypeId element = ParseType(false);
        if (element == 0)
        {
            return false;
        }
        if (kind == Res::Texture2DMS && Accept(","))
        {
            u32 samples = 0;
            if (!ConstantU32(&samples))
            {
                return false;
            }
        }
        if (!Expect(">", "expected '>'"))
        {
            return false;
        }
        if (isTexture && (!IsNumericScalarOrVector(element) || IsBoolish(element)))
        {
            return Fail(start, "texture elements must be numeric scalars or vectors");
        }
        if (kind == Res::CBuffer && T(element).kind != TK::Struct)
        {
            return Fail(start, "ConstantBuffer<T> needs a structure type");
        }
        decl.element = element;
        if (kind == Res::CBuffer)
        {
            decl.structDef = T(element).a;
        }
    }
    else if (kind == Res::StructuredBuffer || kind == Res::RWStructuredBuffer ||
             kind == Res::CBuffer)
    {
        return Fail(start, "this resource type needs an element type");
    }
    decl.name = ExpectIdent("expected a resource name");
    if (m_error)
    {
        return false;
    }
    if (IsPunct("["))
    {
        ++m_cursor;
        if (Accept("]"))
        {
            decl.arraySize = kUnboundedArray;
        }
        else
        {
            u32 length = 0;
            if (!ConstantU32(&length) || !Expect("]", "expected ']'"))
            {
                return false;
            }
            if (length == 0)
            {
                return FailHere("array length must be positive");
            }
            decl.arraySize = length;
        }
        if (IsPunct("["))
        {
            return FailHere("multi-dimensional resource arrays are not supported");
        }
    }
    if (Accept(":"))
    {
        if (!IsIdent("register"))
        {
            return FailHere("expected register(...) after a resource");
        }
        ++m_cursor;
        if (!ParseRegister(&decl))
        {
            return false;
        }
    }
    if (!Expect(";", "expected ';' after a resource declaration"))
    {
        return false;
    }
    if (!m_resources.Push(decl))
    {
        return FailHere("out of memory");
    }
    Symbol symbol{};
    symbol.name = decl.name;
    symbol.kind = SK::Resource;
    symbol.index = m_resources.Size() - 1;
    return Declare(symbol);
}

bool Compiler::ParseCBuffer() noexcept
{
    ++m_cursor;  // cbuffer
    ResourceDecl decl{};
    decl.kind = Res::CBuffer;
    decl.registerClass = RegisterClass::ConstantBuffer;
    decl.cbufferMembersGlobal = true;
    decl.name = ExpectIdent("expected a cbuffer name");
    if (m_error)
    {
        return false;
    }
    if (Accept(":"))
    {
        if (!IsIdent("register"))
        {
            return FailHere("expected register(...) after a cbuffer name");
        }
        ++m_cursor;
        if (!ParseRegister(&decl))
        {
            return false;
        }
    }
    StructDef def{};
    def.name = decl.name;
    if (!m_structs.Push(def))
    {
        return FailHere("out of memory");
    }
    const u32 defIndex = m_structs.Size() - 1;
    if (!ParseStructBody(defIndex))
    {
        return false;
    }
    (void)Accept(";");
    decl.structDef = defIndex;
    decl.element = StructType(defIndex, LayoutLogical, false);
    if (!m_resources.Push(decl))
    {
        return FailHere("out of memory");
    }
    const u32 resourceIndex = m_resources.Size() - 1;
    const StructDef& stored = m_structs[defIndex];
    for (u32 member = 0; member < stored.memberCount; ++member)
    {
        Symbol symbol{};
        symbol.name = m_members[stored.firstMember + member].name;
        symbol.kind = SK::CBufferMember;
        symbol.index = resourceIndex;
        symbol.member = member;
        if (!Declare(symbol))
        {
            return false;
        }
    }
    return true;
}

bool Compiler::ParseGlobalVariable(TypeId type, Name name, bool isStatic, bool isConst,
                                   bool isGroupShared) noexcept
{
    for (;;)
    {
        TypeId varType = type;
        if (!ParseArraySuffix(&varType))
        {
            return false;
        }
        if (Accept(":"))
        {
            return FailHere("semantics and registers on global variables are not supported");
        }
        Symbol symbol{};
        symbol.name = name;
        if (isGroupShared)
        {
            if (IsPunct("="))
            {
                return FailHere("groupshared variables cannot have initializers");
            }
            const u32 id = NewId();
            Inst(m_globals, spv::OpVariable,
                 {SpirvType(PointerTo(spv::Workgroup, varType)), id, spv::Workgroup});
            (void)m_interface.Push(id);
            symbol.kind = SK::Var;
            symbol.value.kind = VK::Ptr;
            symbol.value.id = id;
            symbol.value.type = varType;
            symbol.value.storage = spv::Workgroup;
        }
        else if (isStatic || isConst)
        {
            Val init{};
            if (Accept("="))
            {
                m_atGlobalScope = true;
                init =
                    IsPunct("{") ? ParseInitializer(varType) : Convert(ParseAssignment(), varType);
                if (m_error)
                {
                    return false;
                }
            }
            else if (isConst)
            {
                return FailHere("a constant needs an initializer");
            }
            if (isConst)
            {
                symbol.kind = SK::Const;
                symbol.value = init;
                symbol.value.readOnly = true;
            }
            else
            {
                const u32 id = NewId();
                const u32 initializer = init.kind == VK::R ? init.id : ConstantNull(varType);
                Inst(m_globals, spv::OpVariable,
                     {SpirvType(PointerTo(spv::Private, varType)), id, spv::Private, initializer});
                (void)m_interface.Push(id);
                symbol.kind = SK::Var;
                symbol.value.kind = VK::Ptr;
                symbol.value.id = id;
                symbol.value.type = varType;
                symbol.value.storage = spv::Private;
            }
        }
        else
        {
            return FailHere("global uniforms outside a cbuffer are not supported; declare them in "
                            "a cbuffer with a register");
        }
        if (!Declare(symbol))
        {
            return false;
        }
        if (!Accept(","))
        {
            break;
        }
        name = ExpectIdent("expected a variable name");
        if (m_error)
        {
            return false;
        }
    }
    return Expect(";", "expected ';' after a declaration");
}

bool Compiler::ParseFunction(TypeId returnType, Name name, const u32* numThreads,
                             u32 declToken) noexcept
{
    FuncDecl decl{};
    decl.name = name;
    decl.returnType = returnType;
    decl.firstParam = m_params.Size();
    decl.declToken = declToken;
    if (numThreads != nullptr)
    {
        decl.numThreads[0] = numThreads[0];
        decl.numThreads[1] = numThreads[1];
        decl.numThreads[2] = numThreads[2];
    }
    if (!Expect("(", "expected '('"))
    {
        return false;
    }
    if (IsIdent("void") && IsPunct(")", 1))
    {
        ++m_cursor;
    }
    while (!m_error && !IsPunct(")"))
    {
        Param param{};
        while (Peek().kind == Tok::Ident)
        {
            if (IsIdent("in") || IsIdent("const") || IsIdent("uniform") || IsIdent("linear") ||
                IsIdent("precise"))
            {
            }
            else if (IsIdent("out"))
            {
                param.dir = ParamDir::Out;
            }
            else if (IsIdent("inout"))
            {
                param.dir = ParamDir::InOut;
            }
            else if (IsIdent("nointerpolation"))
            {
                param.interp |= InterpFlat;
            }
            else if (IsIdent("noperspective"))
            {
                param.interp |= InterpNoPerspective;
            }
            else if (IsIdent("centroid"))
            {
                param.interp |= InterpCentroid;
            }
            else if (IsIdent("sample"))
            {
                param.interp |= InterpSample;
            }
            else if (IsIdent("point") || IsIdent("line") || IsIdent("triangle"))
            {
                return FailHere("geometry shader parameters are not supported");
            }
            else
            {
                break;
            }
            ++m_cursor;
        }
        if (ResourceKindOf(Peek()) != Res::None)
        {
            return FailHere("resources as function parameters are not supported by the Vulkan "
                            "HLSL compiler; use the global resource directly");
        }
        param.type = ParseType(false);
        if (param.type == 0)
        {
            return false;
        }
        param.name = ExpectIdent("expected a parameter name");
        if (!ParseArraySuffix(&param.type))
        {
            return false;
        }
        if (Accept(":") && !ParseSemantic(&param.semantic))
        {
            return false;
        }
        if (IsPunct("="))
        {
            return FailHere("default parameter values are not supported");
        }
        if (!m_params.Push(param))
        {
            return FailHere("out of memory");
        }
        ++decl.paramCount;
        if (!Accept(","))
        {
            break;
        }
    }
    if (!Expect(")", "expected ')'"))
    {
        return false;
    }
    if (Accept(":") && !ParseSemantic(&decl.returnSemantic))
    {
        return false;
    }
    if (Accept(";"))
    {
        decl.bodyToken = 0;
    }
    else
    {
        if (!IsPunct("{"))
        {
            return FailHere("expected a function body");
        }
        decl.bodyToken = m_cursor;
        if (!SkipBalanced("{", "}"))
        {
            return false;
        }
    }
    return m_funcs.Push(decl) || FailHere("out of memory");
}

bool Compiler::ParseTopLevel() noexcept
{
    while (!m_error && Peek().kind != Tok::End)
    {
        if (Accept(";"))
        {
            continue;
        }
        u32 numThreads[3] = {0, 0, 0};
        bool hasNumThreads = false;
        while (IsPunct("["))
        {
            ++m_cursor;
            const Name attribute = ExpectIdent("expected an attribute name");
            if (m_error)
            {
                return false;
            }
            if (attribute.Is("numthreads"))
            {
                if (!Expect("(", "expected '('") || !ConstantU32(&numThreads[0]) ||
                    !Expect(",", "expected ','") || !ConstantU32(&numThreads[1]) ||
                    !Expect(",", "expected ','") || !ConstantU32(&numThreads[2]) ||
                    !Expect(")", "expected ')'"))
                {
                    return false;
                }
                hasNumThreads = true;
            }
            else if (IsPunct("("))
            {
                if (!SkipBalanced("(", ")"))
                {
                    return false;
                }
            }
            if (!Expect("]", "expected ']'"))
            {
                return false;
            }
        }
        const u32 declToken = m_cursor;
        if (IsIdent("struct"))
        {
            ++m_cursor;
            StructDef def{};
            def.name = ExpectIdent("expected a structure name");
            if (m_error)
            {
                return false;
            }
            if (!m_structs.Push(def))
            {
                return FailHere("out of memory");
            }
            const u32 defIndex = m_structs.Size() - 1;
            Symbol symbol{};
            symbol.name = def.name;
            symbol.kind = SK::Struct;
            symbol.index = defIndex;
            if (!Declare(symbol) || !ParseStructBody(defIndex))
            {
                return false;
            }
            if (!Expect(";", "expected ';' after a structure"))
            {
                return false;
            }
            continue;
        }
        if (IsIdent("cbuffer"))
        {
            if (!ParseCBuffer())
            {
                return false;
            }
            continue;
        }
        if (IsIdent("tbuffer") || IsIdent("typedef") || IsIdent("namespace") ||
            IsIdent("interface") || IsIdent("class") || IsIdent("template"))
        {
            return FailHere("this declaration is not supported by the Vulkan HLSL compiler");
        }
        bool isStatic = false;
        bool isConst = false;
        bool isGroupShared = false;
        bool isUniform = false;
        while (Peek().kind == Tok::Ident)
        {
            if (IsIdent("static"))
            {
                isStatic = true;
            }
            else if (IsIdent("const"))
            {
                isConst = true;
            }
            else if (IsIdent("groupshared"))
            {
                isGroupShared = true;
            }
            else if (IsIdent("uniform") || IsIdent("extern"))
            {
                isUniform = true;
            }
            else if (IsIdent("inline") || IsIdent("precise") || IsIdent("volatile") ||
                     IsIdent("column_major"))
            {
            }
            else if (IsIdent("row_major"))
            {
                return FailHere("row_major matrices are not supported by the Vulkan HLSL compiler");
            }
            else
            {
                break;
            }
            ++m_cursor;
        }
        const Res resource = ResourceKindOf(Peek());
        if (resource != Res::None)
        {
            if (isStatic || isGroupShared)
            {
                return FailHere("resources cannot be static or groupshared");
            }
            if (!ParseResourceDecl(resource))
            {
                return false;
            }
            continue;
        }
        if (IsIdent("Buffer") || IsIdent("RWBuffer") || IsIdent("AppendStructuredBuffer") ||
            IsIdent("ConsumeStructuredBuffer") || IsIdent("Texture2DMSArray") ||
            IsIdent("RasterizerOrderedBuffer") || IsIdent("RaytracingAccelerationStructure"))
        {
            return FailHere("this resource type is not supported by the Vulkan HLSL compiler");
        }
        const TypeId type = ParseType(true);
        if (type == 0)
        {
            return false;
        }
        const Name name = ExpectIdent("expected a name");
        if (m_error)
        {
            return false;
        }
        if (IsPunct("("))
        {
            if (!ParseFunction(type, name, hasNumThreads ? numThreads : nullptr, declToken))
            {
                return false;
            }
            continue;
        }
        if (T(type).kind == TK::Void)
        {
            return FailHere("a variable cannot be void");
        }
        if (isUniform && (isStatic || isGroupShared))
        {
            return FailHere("conflicting storage qualifiers");
        }
        if (!ParseGlobalVariable(type, name, isStatic, isConst, isGroupShared))
        {
            return false;
        }
        m_atGlobalScope = true;
    }
    return !m_error;
}

// ---- Functions ------------------------------------------------------------------

bool Compiler::CompileFunction(u32 declIndex) noexcept
{
    if (m_error)
    {
        return false;
    }
    if (m_funcs[declIndex].spirvId != 0)
    {
        if (m_funcs[declIndex].compiling)
        {
            const Token& at = m_tokens[m_funcs[declIndex].declToken];
            return Fail(at, "recursion is not allowed in shaders");
        }
        return true;
    }
    const FuncDecl decl = m_funcs[declIndex];
    if (decl.bodyToken == 0)
    {
        return Fail(m_tokens[decl.declToken], "function is declared but never defined");
    }

    TypeId paramTypes[32] = {};
    if (decl.paramCount > 32)
    {
        return Fail(m_tokens[decl.declToken], "functions are limited to 32 parameters");
    }
    for (u32 index = 0; index < decl.paramCount; ++index)
    {
        const Param& param = m_params[decl.firstParam + index];
        paramTypes[index] =
            param.dir == ParamDir::In ? param.type : PointerTo(spv::Function, param.type);
    }
    const TypeId functionType = FunctionType(decl.returnType, paramTypes, decl.paramCount);
    const u32 functionId = NewId();
    m_funcs[declIndex].spirvId = functionId;
    m_funcs[declIndex].functionType = functionType;
    m_funcs[declIndex].compiling = true;

    void* storage = m_arena.Allocate(sizeof(FuncState), alignof(FuncState));
    if (storage == nullptr)
    {
        return FailHere("out of memory");
    }
    FuncState* state = new (storage) FuncState();
    state->variables.Init(&m_arena);
    state->body.Init(&m_arena);
    state->loops.Init(&m_arena);
    state->returnType = decl.returnType;
    state->blockOpen = true;

    const u32 savedCursor = m_cursor;
    const bool savedGlobal = m_atGlobalScope;
    const bool savedPrecise = m_precise;
    // A callee sees only globals: hide the caller's locals.
    const u32 savedSymbolCount = m_symbols.Size();
    Vec<Symbol> hidden;
    hidden.Init(&m_arena);
    for (u32 index = m_globalSymbolCount; index < savedSymbolCount; ++index)
    {
        (void)hidden.Push(m_symbols[index]);
    }
    Vec<u32> hiddenScopes;
    hiddenScopes.Init(&m_arena);
    for (u32 index = 0; index < m_scopes.Size(); ++index)
    {
        (void)hiddenScopes.Push(m_scopes[index]);
    }
    m_symbols.Shrink(m_globalSymbolCount);
    m_scopes.Clear();

    (void)m_funcStack.Push(state);
    m_atGlobalScope = false;
    m_precise = false;
    PushScope();

    Vec<u32> header;
    header.Init(&m_arena);
    Inst(header, spv::OpFunction,
         {SpirvType(decl.returnType), functionId, 0, SpirvType(functionType)});
    const u32 entryLabel = NewId();
    for (u32 index = 0; index < decl.paramCount && !m_error; ++index)
    {
        const Param param = m_params[decl.firstParam + index];
        const u32 paramId = NewId();
        Inst(header, spv::OpFunctionParameter, {SpirvType(paramTypes[index]), paramId});
        Symbol symbol{};
        symbol.name = param.name;
        symbol.kind = SK::Var;
        symbol.value.kind = VK::Ptr;
        symbol.value.type = param.type;
        symbol.value.storage = spv::Function;
        if (param.dir == ParamDir::In)
        {
            const u32 local = NewLocal(param.type);
            Inst(state->body, spv::OpStore, {local, paramId});
            symbol.value.id = local;
        }
        else
        {
            symbol.value.id = paramId;
        }
        (void)Declare(symbol);
    }
    m_cursor = decl.bodyToken;
    (void)ParseBlock();
    if (!m_error && state->blockOpen)
    {
        if (T(decl.returnType).kind == TK::Void)
        {
            Terminate(spv::OpReturn);
        }
        else
        {
            Terminate(spv::OpUnreachable);
        }
    }
    PopScope();
    m_funcStack.Shrink(m_funcStack.Size() - 1);

    for (u32 index = 0; index < header.Size(); ++index)
    {
        (void)m_functions.Push(header[index]);
    }
    Inst(m_functions, spv::OpLabel, {entryLabel});
    for (u32 index = 0; index < state->variables.Size(); ++index)
    {
        (void)m_functions.Push(state->variables[index]);
    }
    for (u32 index = 0; index < state->body.Size(); ++index)
    {
        (void)m_functions.Push(state->body[index]);
    }
    Inst(m_functions, spv::OpFunctionEnd, {});

    // Restore the caller.
    m_symbols.Shrink(m_globalSymbolCount);
    for (u32 index = 0; index < hidden.Size(); ++index)
    {
        (void)m_symbols.Push(hidden[index]);
    }
    m_scopes.Clear();
    for (u32 index = 0; index < hiddenScopes.Size(); ++index)
    {
        (void)m_scopes.Push(hiddenScopes[index]);
    }
    m_cursor = savedCursor;
    m_atGlobalScope = savedGlobal;
    m_precise = savedPrecise;
    m_funcs[declIndex].compiling = false;
    return !m_error;
}

// ---- Statements -----------------------------------------------------------------

bool Compiler::ParseBlock() noexcept
{
    if (!Expect("{", "expected '{'"))
    {
        return false;
    }
    PushScope();
    while (!m_error && !IsPunct("}"))
    {
        if (Peek().kind == Tok::End)
        {
            return FailHere("unterminated block");
        }
        if (!ParseStatement())
        {
            break;
        }
    }
    PopScope();
    return Expect("}", "expected '}'");
}

bool Compiler::ParseStatement() noexcept
{
    if (m_depth >= 64)
    {
        return FailHere("statements are nested too deeply");
    }
    ++m_depth;
    const bool ok = ParseStatementInner();
    --m_depth;
    return ok;
}

bool Compiler::ParseStatementInner() noexcept
{
    if (m_error)
    {
        return false;
    }
    u32 loopControl = 0;
    while (IsPunct("[") && Peek(1).kind == Tok::Ident)
    {
        ++m_cursor;
        const Name attribute = ExpectIdent("expected an attribute");
        if (attribute.Is("unroll"))
        {
            loopControl = 1;
        }
        else if (attribute.Is("loop"))
        {
            loopControl = 2;
        }
        if (IsPunct("(") && !SkipBalanced("(", ")"))
        {
            return false;
        }
        if (!Expect("]", "expected ']'"))
        {
            return false;
        }
    }
    if (IsPunct("{"))
    {
        return ParseBlock();
    }
    if (Accept(";"))
    {
        return true;
    }
    if (IsIdent("if"))
    {
        return ParseIf();
    }
    if (IsIdent("for"))
    {
        return ParseFor(loopControl);
    }
    if (IsIdent("while"))
    {
        return ParseWhile(loopControl);
    }
    if (IsIdent("do"))
    {
        return ParseDoWhile(loopControl);
    }
    if (IsIdent("switch"))
    {
        return FailHere("switch statements are not supported by the Vulkan HLSL compiler");
    }
    if (IsIdent("return"))
    {
        return ParseReturn();
    }
    if (IsIdent("break") || IsIdent("continue"))
    {
        const bool isBreak = IsIdent("break");
        const Token& at = Next();
        FuncState& state = *m_funcStack.Back();
        if (state.loops.Size() == 0)
        {
            return Fail(at, "break or continue outside a loop");
        }
        Branch(isBreak ? state.loops.Back().merge : state.loops.Back().continueTarget);
        return Expect(";", "expected ';'");
    }
    if (IsIdent("discard"))
    {
        const Token& at = Next();
        if (m_desc.stage != ShaderStage::Pixel)
        {
            return Fail(at, "discard is only valid in a pixel shader");
        }
        Terminate(spv::OpKill);
        return Expect(";", "expected ';'");
    }
    // Declarations start with a qualifier, or a type followed by a name.
    bool declaration = false;
    for (const char* qualifier : kQualifiers)
    {
        if (IsIdent(qualifier))
        {
            declaration = true;
        }
    }
    if (!declaration && IsTypeStart())
    {
        declaration = IsIdent("vector") || IsIdent("matrix") || Peek(1).kind == Tok::Ident;
    }
    if (declaration)
    {
        return ParseLocalDeclaration();
    }
    (void)ParseExpression();
    return Expect(";", "expected ';' after an expression");
}

bool Compiler::ParseLocalDeclaration() noexcept
{
    bool isConst = false;
    bool isStatic = false;
    bool isPrecise = false;
    while (Peek().kind == Tok::Ident)
    {
        if (IsIdent("const"))
        {
            isConst = true;
        }
        else if (IsIdent("static"))
        {
            isStatic = true;
        }
        else if (IsIdent("precise"))
        {
            isPrecise = true;
        }
        else if (IsIdent("uniform") || IsIdent("volatile") || IsIdent("column_major") ||
                 IsIdent("linear"))
        {
        }
        else if (IsIdent("groupshared") || IsIdent("row_major") || IsIdent("in") ||
                 IsIdent("out") || IsIdent("inout"))
        {
            return FailHere("this qualifier is not valid on a local variable");
        }
        else
        {
            break;
        }
        ++m_cursor;
    }
    const TypeId base = ParseType(false);
    if (base == 0)
    {
        return false;
    }
    for (;;)
    {
        const Name name = ExpectIdent("expected a variable name");
        TypeId type = base;
        if (m_error || !ParseArraySuffix(&type))
        {
            return false;
        }
        Symbol symbol{};
        symbol.name = name;
        symbol.kind = SK::Var;
        symbol.precise = isPrecise;
        if (isStatic)
        {
            // A static local is a private global that is initialised once.
            Val init{};
            if (Accept("="))
            {
                const bool wasGlobal = m_atGlobalScope;
                m_atGlobalScope = true;
                init = IsPunct("{") ? ParseInitializer(type) : Convert(ParseAssignment(), type);
                m_atGlobalScope = wasGlobal;
                if (m_error)
                {
                    return false;
                }
            }
            const u32 id = NewId();
            Inst(m_globals, spv::OpVariable,
                 {SpirvType(PointerTo(spv::Private, type)), id, spv::Private,
                  init.kind == VK::R ? init.id : ConstantNull(type)});
            (void)m_interface.Push(id);
            symbol.value.kind = VK::Ptr;
            symbol.value.id = id;
            symbol.value.type = type;
            symbol.value.storage = spv::Private;
            symbol.value.readOnly = isConst;
            if (!Declare(symbol))
            {
                return false;
            }
        }
        else
        {
            Val init{};
            const bool hasInit = Accept("=");
            if (hasInit)
            {
                const bool wasPrecise = m_precise;
                m_precise = isPrecise;
                init = IsPunct("{") ? ParseInitializer(type) : ParseAssignment();
                m_precise = wasPrecise;
                if (m_error)
                {
                    return false;
                }
            }
            else if (isConst)
            {
                return FailHere("a constant needs an initializer");
            }
            if (isConst && init.kind == VK::R && init.isConst && IsScalar(type))
            {
                symbol.kind = SK::Const;
                symbol.value = Convert(init, type);
                symbol.value.readOnly = true;
            }
            else
            {
                symbol.value.kind = VK::Ptr;
                symbol.value.id = NewLocal(type);
                symbol.value.type = type;
                symbol.value.storage = spv::Function;
                if (hasInit && !Store(symbol.value, init))
                {
                    return false;
                }
                symbol.value.readOnly = isConst;
            }
            // Declared after the initialiser: `float x = x;` reads an outer `x`.
            if (!Declare(symbol))
            {
                return false;
            }
        }
        if (!Accept(","))
        {
            break;
        }
    }
    return Expect(";", "expected ';' after a declaration");
}

bool Compiler::ParseIf() noexcept
{
    const Token& at = Next();  // if
    if (!Expect("(", "expected '(' after if"))
    {
        return false;
    }
    const Val condition = ToBool(ParseExpression());
    if (m_error || !Expect(")", "expected ')'"))
    {
        return false;
    }
    if (!IsScalar(condition.type))
    {
        return Fail(at, "an if condition must be a scalar");
    }
    const u32 thenLabel = NewId();
    const u32 elseLabel = NewId();
    const u32 mergeLabel = NewId();
    OpVoid(spv::OpSelectionMerge, {mergeLabel, 0});
    Terminate(spv::OpBranchConditional, {condition.id, thenLabel, elseLabel});
    Label(thenLabel);
    PushScope();
    const bool thenOk = ParseStatement();
    PopScope();
    if (!thenOk)
    {
        return false;
    }
    Branch(mergeLabel);
    Label(elseLabel);
    if (IsIdent("else"))
    {
        ++m_cursor;
        PushScope();
        const bool elseOk = ParseStatement();
        PopScope();
        if (!elseOk)
        {
            return false;
        }
    }
    Branch(mergeLabel);
    Label(mergeLabel);
    return !m_error;
}

bool Compiler::ParseFor(u32 loopControl) noexcept
{
    ++m_cursor;  // for
    if (!Expect("(", "expected '(' after for"))
    {
        return false;
    }
    PushScope();
    if (!Accept(";"))
    {
        if (!ParseStatement())  // a declaration or expression, with its ';'
        {
            PopScope();
            return false;
        }
    }
    FuncState& state = *m_funcStack.Back();
    const u32 header = NewId();
    const u32 body = NewId();
    const u32 continueLabel = NewId();
    const u32 merge = NewId();
    const u32 conditionLabel = NewId();
    Branch(header);
    Label(header);
    // The header holds only the merge instruction, so a condition that contains
    // control flow of its own (a short-circuit) stays structured.
    OpVoid(spv::OpLoopMerge, {merge, continueLabel, loopControl});
    Terminate(spv::OpBranch, {conditionLabel});
    Label(conditionLabel);
    Val condition{};
    const Token& conditionToken = Peek();
    const bool hasCondition = !IsPunct(";");
    if (hasCondition)
    {
        condition = ToBool(ParseExpression());
        if (m_error)
        {
            PopScope();
            return false;
        }
        if (!IsScalar(condition.type))
        {
            PopScope();
            return Fail(conditionToken, "a loop condition must be a scalar");
        }
    }
    if (!Expect(";", "expected ';' in for"))
    {
        PopScope();
        return false;
    }
    const u32 incrementStart = m_cursor;
    u32 depth = 0;
    while (!m_error)
    {
        if (Peek().kind == Tok::End)
        {
            PopScope();
            return FailHere("unterminated for");
        }
        if (IsPunct("("))
        {
            ++depth;
        }
        else if (IsPunct(")"))
        {
            if (depth == 0)
            {
                break;
            }
            --depth;
        }
        ++m_cursor;
    }
    ++m_cursor;  // ')'
    if (hasCondition)
    {
        Terminate(spv::OpBranchConditional, {condition.id, body, merge});
    }
    else
    {
        Terminate(spv::OpBranch, {body});
    }
    Label(body);
    (void)state.loops.Push(LoopTargets{merge, continueLabel});
    PushScope();
    const bool bodyOk = ParseStatement();
    PopScope();
    state.loops.Shrink(state.loops.Size() - 1);
    if (!bodyOk)
    {
        PopScope();
        return false;
    }
    const u32 afterBody = m_cursor;
    Branch(continueLabel);
    Label(continueLabel);
    m_cursor = incrementStart;
    while (!m_error && !IsPunct(")"))
    {
        (void)ParseAssignment();
        if (!Accept(","))
        {
            break;
        }
    }
    if (!m_error && !IsPunct(")"))
    {
        PopScope();
        return FailHere("expected ')' after the loop increment");
    }
    m_cursor = afterBody;
    Branch(header);
    Label(merge);
    PopScope();
    return !m_error;
}

bool Compiler::ParseWhile(u32 loopControl) noexcept
{
    ++m_cursor;  // while
    FuncState& state = *m_funcStack.Back();
    const u32 header = NewId();
    const u32 body = NewId();
    const u32 continueLabel = NewId();
    const u32 merge = NewId();
    const u32 conditionLabel = NewId();
    Branch(header);
    Label(header);
    OpVoid(spv::OpLoopMerge, {merge, continueLabel, loopControl});
    Terminate(spv::OpBranch, {conditionLabel});
    Label(conditionLabel);
    if (!Expect("(", "expected '(' after while"))
    {
        return false;
    }
    const Token& at = Peek();
    const Val condition = ToBool(ParseExpression());
    if (m_error || !Expect(")", "expected ')'"))
    {
        return false;
    }
    if (!IsScalar(condition.type))
    {
        return Fail(at, "a loop condition must be a scalar");
    }
    Terminate(spv::OpBranchConditional, {condition.id, body, merge});
    Label(body);
    (void)state.loops.Push(LoopTargets{merge, continueLabel});
    PushScope();
    const bool ok = ParseStatement();
    PopScope();
    state.loops.Shrink(state.loops.Size() - 1);
    if (!ok)
    {
        return false;
    }
    Branch(continueLabel);
    Label(continueLabel);
    Branch(header);
    Label(merge);
    return !m_error;
}

bool Compiler::ParseDoWhile(u32 loopControl) noexcept
{
    ++m_cursor;  // do
    FuncState& state = *m_funcStack.Back();
    const u32 header = NewId();
    const u32 body = NewId();
    const u32 continueLabel = NewId();
    const u32 merge = NewId();
    Branch(header);
    Label(header);
    OpVoid(spv::OpLoopMerge, {merge, continueLabel, loopControl});
    Terminate(spv::OpBranch, {body});
    Label(body);
    (void)state.loops.Push(LoopTargets{merge, continueLabel});
    PushScope();
    const bool ok = ParseStatement();
    PopScope();
    state.loops.Shrink(state.loops.Size() - 1);
    if (!ok)
    {
        return false;
    }
    Branch(continueLabel);
    Label(continueLabel);
    if (!IsIdent("while"))
    {
        return FailHere("expected while after a do body");
    }
    ++m_cursor;
    if (!Expect("(", "expected '('"))
    {
        return false;
    }
    const Token& at = Peek();
    const Val condition = ToBool(ParseExpression());
    if (m_error || !Expect(")", "expected ')'") || !Expect(";", "expected ';'"))
    {
        return false;
    }
    if (!IsScalar(condition.type))
    {
        return Fail(at, "a loop condition must be a scalar");
    }
    Terminate(spv::OpBranchConditional, {condition.id, header, merge});
    Label(merge);
    return !m_error;
}

bool Compiler::ParseReturn() noexcept
{
    const Token& at = Next();  // return
    FuncState& state = *m_funcStack.Back();
    if (T(state.returnType).kind == TK::Void)
    {
        if (!IsPunct(";"))
        {
            return Fail(at, "a void function cannot return a value");
        }
        Terminate(spv::OpReturn);
        return Expect(";", "expected ';'");
    }
    if (IsPunct(";"))
    {
        return Fail(at, "this function must return a value");
    }
    const Val value = Convert(ParseExpression(), state.returnType);
    if (m_error)
    {
        return false;
    }
    Terminate(spv::OpReturnValue, {value.id});
    return Expect(";", "expected ';'");
}

// ---- Entry point ----------------------------------------------------------------

namespace
{

void UpperSemantic(const Semantic& semantic, char* out, usize capacity) noexcept
{
    usize length = semantic.name.length < capacity - 1 ? semantic.name.length : capacity - 1;
    for (usize index = 0; index < length; ++index)
    {
        out[index] = Upper(semantic.name.text[index]);
    }
    out[length] = 0;
}

}  // namespace

bool Compiler::EmitEntryPoint(u32 declIndex) noexcept
{
    const FuncDecl decl = m_funcs[declIndex];
    const Token& at = m_tokens[decl.declToken];
    const ShaderStage stage = m_desc.stage;
    if (stage == ShaderStage::Compute)
    {
        if (decl.numThreads[0] == 0 || decl.numThreads[1] == 0 || decl.numThreads[2] == 0)
        {
            return Fail(at, "a compute entry point needs [numthreads(x, y, z)]");
        }
        m_out->localSize[0] = decl.numThreads[0];
        m_out->localSize[1] = decl.numThreads[1];
        m_out->localSize[2] = decl.numThreads[2];
    }
    if (!CompileFunction(declIndex))
    {
        return false;
    }
    const u32 userFunction = m_funcs[declIndex].spirvId;

    void* storage = m_arena.Allocate(sizeof(FuncState), alignof(FuncState));
    if (storage == nullptr)
    {
        return Fail(at, "out of memory");
    }
    FuncState* state = new (storage) FuncState();
    state->variables.Init(&m_arena);
    state->body.Init(&m_arena);
    state->loops.Init(&m_arena);
    state->returnType = VoidType();
    state->blockOpen = true;
    (void)m_funcStack.Push(state);
    m_atGlobalScope = false;

    u32 nextInputLocation = 0;
    u32 nextOutputLocation = 0;
    bool depthReplacing = false;

    // Creates an interface variable and returns its id.
    const auto makeVar = [&](u32 storageClass, TypeId type, bool builtin, u32 value,
                             u8 interp) noexcept -> u32
    {
        const u32 id = NewId();
        Inst(m_globals, spv::OpVariable,
             {SpirvType(PointerTo(storageClass, type)), id, storageClass});
        (void)m_interface.Push(id);
        if (builtin)
        {
            Decorate(id, spv::DecBuiltIn, {value});
        }
        else
        {
            Decorate(id, spv::DecLocation, {value});
            if (storageClass == spv::Input && stage == ShaderStage::Pixel)
            {
                if ((interp & InterpFlat) != 0 || IsInteger(type) || IsBoolish(type))
                {
                    Decorate(id, spv::DecFlat);
                }
                if ((interp & InterpNoPerspective) != 0)
                {
                    Decorate(id, spv::DecNoPerspective);
                }
                if ((interp & InterpCentroid) != 0)
                {
                    Decorate(id, spv::DecCentroid);
                }
            }
        }
        return id;
    };

    // Reads one input with a semantic as a value of `type`.
    const auto readInput = [&](TypeId type, const Semantic& semantic, u8 interp) noexcept -> Val
    {
        if (!semantic.present)
        {
            (void)Fail(at, "every entry-point input needs a semantic");
            return Val{};
        }
        if ((interp & InterpSample) != 0)
        {
            (void)Fail(at, "sample interpolation is not supported by the Vulkan HLSL compiler");
            return Val{};
        }
        Val raw{};
        raw.kind = VK::R;
        const TypeId uint3 = VectorOf(UintType(), 3);
        u32 builtin = 0xFFFFFFFFU;
        TypeId builtinType = 0;
        if (stage == ShaderStage::Compute)
        {
            if (SemanticIs(semantic, "SV_DISPATCHTHREADID"))
            {
                builtin = spv::BuiltInGlobalInvocationId;
                builtinType = uint3;
            }
            else if (SemanticIs(semantic, "SV_GROUPID"))
            {
                builtin = spv::BuiltInWorkgroupId;
                builtinType = uint3;
            }
            else if (SemanticIs(semantic, "SV_GROUPTHREADID"))
            {
                builtin = spv::BuiltInLocalInvocationId;
                builtinType = uint3;
            }
            else if (SemanticIs(semantic, "SV_GROUPINDEX"))
            {
                builtin = spv::BuiltInLocalInvocationIndex;
                builtinType = UintType();
            }
            else
            {
                (void)Fail(at, "unknown compute-shader input semantic");
                return Val{};
            }
        }
        else if (stage == ShaderStage::Vertex)
        {
            if (SemanticIs(semantic, "SV_VERTEXID"))
            {
                builtin = spv::BuiltInVertexIndex;
                builtinType = IntType();
            }
            else if (SemanticIs(semantic, "SV_INSTANCEID"))
            {
                // Vulkan's InstanceIndex starts at firstInstance; HLSL's SV_InstanceID
                // starts at 0, so the base is subtracted. Without shaderDrawParameters
                // the base is unavailable and the value would be wrong for any draw
                // with a first instance, so the shader is refused instead.
                if (!m_desc.features.drawParameters)
                {
                    (void)Fail(at, "SV_InstanceID needs the shaderDrawParameters device feature "
                                   "on Vulkan");
                    return Val{};
                }
                const u32 index =
                    makeVar(spv::Input, IntType(), true, spv::BuiltInInstanceIndex, 0);
                raw.type = IntType();
                raw.id = Op(spv::OpLoad, IntType(), {index});
                Capability(spv::CapDrawParameters);
                const u32 base = makeVar(spv::Input, IntType(), true, spv::BuiltInBaseInstance, 0);
                const u32 baseValue = Op(spv::OpLoad, IntType(), {base});
                raw.id = Op(spv::OpISub, IntType(), {raw.id, baseValue});
                return Convert(raw, type);
            }
            else if (SemanticIs(semantic, "TEXCOORD"))
            {
                if (!IsNumericScalarOrVector(type) || IsBoolish(type))
                {
                    (void)Fail(at, "vertex inputs must be numeric scalars or vectors");
                    return Val{};
                }
                const u32 var = makeVar(spv::Input, type, false, semantic.index, 0);
                raw.type = type;
                raw.id = Op(spv::OpLoad, type, {var});
                return raw;
            }
            else
            {
                (void)Fail(at, "vertex inputs must use TEXCOORD<n> semantics: the RHI binds "
                               "attribute <n> as TEXCOORD<n> on every backend");
                return Val{};
            }
        }
        else
        {
            if (SemanticIs(semantic, "SV_POSITION"))
            {
                const TypeId float4 = VectorOf(FloatType(), 4);
                const u32 var = makeVar(spv::Input, float4, true, spv::BuiltInFragCoord, 0);
                const u32 coord = Op(spv::OpLoad, float4, {var});
                // D3D reports clip w in SV_Position.w; FragCoord.w is 1 / w.
                const u32 w = Op(spv::OpCompositeExtract, FloatType(), {coord, 3});
                const u32 inverse = Op(spv::OpFDiv, FloatType(), {ConstFloat(1.0F), w});
                raw.type = float4;
                raw.id = Op(spv::OpCompositeInsert, float4, {inverse, coord, 3});
                return Convert(raw, type);
            }
            if (SemanticIs(semantic, "SV_ISFRONTFACE"))
            {
                builtin = spv::BuiltInFrontFacing;
                builtinType = BoolType();
            }
            else if (SemanticIs(semantic, "SV_PRIMITIVEID"))
            {
                builtin = spv::BuiltInPrimitiveId;
                builtinType = IntType();
            }
            else if (semantic.name.length >= 3 && Upper(semantic.name.text[0]) == 'S' &&
                     Upper(semantic.name.text[1]) == 'V' && semantic.name.text[2] == '_')
            {
                (void)Fail(at, "this pixel-shader system value is not supported by the Vulkan "
                               "HLSL compiler");
                return Val{};
            }
            else
            {
                if (!IsNumericScalarOrVector(type))
                {
                    (void)Fail(at, "stage varyings must be scalars or vectors");
                    return Val{};
                }
                char name[32] = {};
                UpperSemantic(semantic, name, sizeof(name));
                u32 location = 0xFFFFFFFFU;
                if (m_desc.linkedInputs != nullptr)
                {
                    for (u32 index = 0; index < m_desc.linkedInputCount; ++index)
                    {
                        const Varying& varying = m_desc.linkedInputs[index];
                        if (varying.index == semantic.index &&
                            std::strcmp(varying.semantic, name) == 0)
                        {
                            location = varying.location;
                        }
                    }
                    if (location == 0xFFFFFFFFU)
                    {
                        (void)Fail(at, "a pixel-shader input has no matching vertex-shader output");
                        return Val{};
                    }
                }
                else
                {
                    location = nextInputLocation++;
                }
                const u32 var = makeVar(spv::Input, type, false, location, interp);
                raw.type = type;
                raw.id = Op(spv::OpLoad, type, {var});
                return raw;
            }
        }
        const u32 var = makeVar(spv::Input, builtinType, true, builtin, 0);
        raw.type = builtinType;
        raw.id = Op(spv::OpLoad, builtinType, {var});
        return Convert(raw, type);
    };

    // Writes one output value with a semantic.
    const auto writeOutput = [&](const Val& value, const Semantic& semantic) noexcept -> bool
    {
        if (!semantic.present)
        {
            return Fail(at, "every entry-point output needs a semantic");
        }
        if (stage == ShaderStage::Vertex)
        {
            if (SemanticIs(semantic, "SV_POSITION"))
            {
                const TypeId float4 = VectorOf(FloatType(), 4);
                const Val converted = Convert(value, float4);
                const u32 var = makeVar(spv::Output, float4, true, spv::BuiltInPosition, 0);
                OpVoid(spv::OpStore, {var, converted.id});
                return !m_error;
            }
            if (semantic.name.length >= 3 && Upper(semantic.name.text[0]) == 'S' &&
                Upper(semantic.name.text[1]) == 'V' && semantic.name.text[2] == '_')
            {
                return Fail(at, "this vertex-shader system value is not supported by the Vulkan "
                                "HLSL compiler");
            }
            if (!IsNumericScalarOrVector(value.type))
            {
                return Fail(at, "stage varyings must be scalars or vectors");
            }
            if (m_out->outputCount >= kMaxVaryings)
            {
                return Fail(at, "too many vertex-shader outputs");
            }
            Varying& varying = m_out->outputs[m_out->outputCount++];
            UpperSemantic(semantic, varying.semantic, sizeof(varying.semantic));
            varying.index = semantic.index;
            varying.location = nextOutputLocation++;
            const u32 var = makeVar(spv::Output, value.type, false, varying.location, 0);
            OpVoid(spv::OpStore, {var, value.id});
            return !m_error;
        }
        if (stage == ShaderStage::Pixel)
        {
            if (SemanticIs(semantic, "SV_TARGET"))
            {
                if (!IsNumericScalarOrVector(value.type))
                {
                    return Fail(at, "SV_Target must be a scalar or vector");
                }
                const u32 var = makeVar(spv::Output, value.type, false, semantic.index, 0);
                OpVoid(spv::OpStore, {var, value.id});
                return !m_error;
            }
            if (SemanticIs(semantic, "SV_DEPTH"))
            {
                const Val converted = Convert(value, FloatType());
                const u32 var = makeVar(spv::Output, FloatType(), true, spv::BuiltInFragDepth, 0);
                OpVoid(spv::OpStore, {var, converted.id});
                depthReplacing = true;
                return !m_error;
            }
            return Fail(at, "unsupported pixel-shader output semantic");
        }
        return Fail(at, "compute shaders have no outputs");
    };

    // Inputs: flatten structure parameters, call, then write outputs.
    u32 args[32] = {};
    u32 outTemps[32] = {};
    for (u32 index = 0; index < decl.paramCount && !m_error; ++index)
    {
        const Param param = m_params[decl.firstParam + index];
        if (param.dir == ParamDir::InOut)
        {
            return Fail(at, "inout entry-point parameters are not supported");
        }
        if (param.dir == ParamDir::Out)
        {
            outTemps[index] = NewLocal(param.type);
            args[index] = outTemps[index];
            continue;
        }
        if (T(param.type).kind == TK::Struct)
        {
            const StructDef& def = m_structs[T(param.type).a];
            u32 parts[64] = {};
            for (u32 member = 0; member < def.memberCount; ++member)
            {
                const StructMember info = m_members[def.firstMember + member];
                if (T(info.type).kind == TK::Struct)
                {
                    return Fail(at, "nested structures in entry-point inputs are not supported");
                }
                parts[member] = readInput(info.type, info.semantic, info.interp | param.interp).id;
            }
            args[index] = OpN(spv::OpCompositeConstruct, param.type, parts, def.memberCount);
        }
        else
        {
            args[index] = readInput(param.type, param.semantic, param.interp).id;
        }
    }
    if (m_error)
    {
        return false;
    }
    u32 callOperands[33] = {userFunction};
    for (u32 index = 0; index < decl.paramCount; ++index)
    {
        callOperands[index + 1] = args[index];
    }
    Val result{};
    result.kind = VK::R;
    result.type = decl.returnType;
    result.id = OpN(spv::OpFunctionCall, decl.returnType, callOperands, decl.paramCount + 1);

    const auto writeValue = [&](const Val& value, const Semantic& semantic) noexcept -> bool
    {
        if (T(value.type).kind == TK::Struct)
        {
            const StructDef& def = m_structs[T(value.type).a];
            for (u32 member = 0; member < def.memberCount; ++member)
            {
                const StructMember info = m_members[def.firstMember + member];
                if (T(info.type).kind == TK::Struct)
                {
                    return Fail(at, "nested structures in entry-point outputs are not supported");
                }
                Val part{};
                part.kind = VK::R;
                part.type = info.type;
                part.id = Op(spv::OpCompositeExtract, info.type, {value.id, member});
                if (!writeOutput(part, info.semantic))
                {
                    return false;
                }
            }
            return true;
        }
        return writeOutput(value, semantic);
    };
    if (T(decl.returnType).kind != TK::Void && !writeValue(result, decl.returnSemantic))
    {
        return false;
    }
    for (u32 index = 0; index < decl.paramCount; ++index)
    {
        const Param param = m_params[decl.firstParam + index];
        if (param.dir != ParamDir::Out)
        {
            continue;
        }
        Val value{};
        value.kind = VK::R;
        value.type = param.type;
        value.id = Op(spv::OpLoad, param.type, {outTemps[index]});
        if (!writeValue(value, param.semantic))
        {
            return false;
        }
    }
    Terminate(spv::OpReturn);
    m_funcStack.Shrink(m_funcStack.Size() - 1);
    if (m_error)
    {
        return false;
    }

    const u32 wrapper = NewId();
    const TypeId voidFunction = FunctionType(VoidType(), nullptr, 0);
    Inst(m_functions, spv::OpFunction,
         {SpirvType(VoidType()), wrapper, 0, SpirvType(voidFunction)});
    Inst(m_functions, spv::OpLabel, {NewId()});
    for (u32 index = 0; index < state->variables.Size(); ++index)
    {
        (void)m_functions.Push(state->variables[index]);
    }
    for (u32 index = 0; index < state->body.Size(); ++index)
    {
        (void)m_functions.Push(state->body[index]);
    }
    Inst(m_functions, spv::OpFunctionEnd, {});

    u32 model = 0;
    switch (stage)
    {
    case ShaderStage::Vertex:
        model = 0;
        break;
    case ShaderStage::Pixel:
        model = 4;
        break;
    case ShaderStage::Compute:
        model = 5;
        break;
    default:
        return Fail(at, "unsupported shader stage");
    }
    const char* entryName = m_desc.entryPoint != nullptr ? m_desc.entryPoint : "main";
    InstString(m_entryPoints, spv::OpEntryPoint, {model, wrapper}, entryName,
               static_cast<u32>(std::strlen(entryName)), m_interface.Data(), m_interface.Size());
    if (stage == ShaderStage::Pixel)
    {
        Inst(m_executionModes, spv::OpExecutionMode, {wrapper, 7});  // OriginUpperLeft
        if (depthReplacing)
        {
            Inst(m_executionModes, spv::OpExecutionMode, {wrapper, 12});
        }
    }
    else if (stage == ShaderStage::Compute)
    {
        Inst(m_executionModes, spv::OpExecutionMode,
             {wrapper, 17, decl.numThreads[0], decl.numThreads[1], decl.numThreads[2]});
    }
    return true;
}

// ---- Driver ---------------------------------------------------------------------

bool Compiler::Run() noexcept
{
    if (m_desc.source == nullptr || m_desc.sourceBytes == 0)
    {
        Token none{};
        return Fail(none, "empty shader source");
    }
    if (!Lex())
    {
        return false;
    }
    m_glsl = NewId();
    Capability(spv::CapShader);
    m_atGlobalScope = true;
    if (!ParseTopLevel())
    {
        return false;
    }
    m_globalSymbolCount = m_symbols.Size();
    const char* entryName = m_desc.entryPoint != nullptr ? m_desc.entryPoint : "main";
    const Name entry{entryName, static_cast<u32>(std::strlen(entryName))};
    u32 found = 0xFFFFFFFFU;
    for (u32 index = 0; index < m_funcs.Size(); ++index)
    {
        if (m_funcs[index].name.Equals(entry) && m_funcs[index].bodyToken != 0)
        {
            if (found != 0xFFFFFFFFU)
            {
                return Fail(m_tokens[m_funcs[index].declToken], "the entry point is overloaded");
            }
            found = index;
        }
    }
    if (found == 0xFFFFFFFFU)
    {
        Token none{};
        return Fail(none, "the entry point function is not defined");
    }
    if (!EmitEntryPoint(found))
    {
        return false;
    }
    if (m_arena.Failed())
    {
        Token none{};
        return Fail(none, "out of memory");
    }

    // Assemble the module in the order the specification requires.
    cyber::memory::DynamicArray<u32>& words = m_out->words;
    words.Clear();
    Vec<u32> preamble;
    preamble.Init(&m_arena);
    InstString(preamble, spv::OpExtInstImport, {m_glsl}, "GLSL.std.450", 12, nullptr, 0);
    Inst(preamble, spv::OpMemoryModel, {0, 1});  // Logical, GLSL450
    const Vec<u32>* sections[] = {&m_capabilities,   &preamble,   &m_entryPoints,
                                  &m_executionModes, &m_debug,    &m_annotations,
                                  &m_globals,        &m_functions};
    usize total = 5;
    for (const Vec<u32>* section : sections)
    {
        total += section->Size();
    }
    if (!words.TryReserve(total))
    {
        Token none{};
        return Fail(none, "out of memory");
    }
    (void)words.TryPushBack(0x07230203U);
    (void)words.TryPushBack(0x00010500U);  // SPIR-V 1.5
    (void)words.TryPushBack(0);
    (void)words.TryPushBack(m_nextId);
    (void)words.TryPushBack(0);
    for (const Vec<u32>* section : sections)
    {
        for (u32 index = 0; index < section->Size(); ++index)
        {
            (void)words.TryPushBack((*section)[index]);
        }
    }
    return true;
}

cyber::core::Status<cyber::core::Error>
Compile(const CompileDesc& desc, cyber::memory::Allocator& allocator, CompileOutput* out) noexcept
{
    using cyber::core::Error;
    using cyber::core::ErrorCode;
    using StatusType = cyber::core::Status<Error>;
    if (out == nullptr)
    {
        return StatusType::Err(Error{ErrorCode::InvalidArgument, "hlsl: no output"});
    }
    out->diagnostic[0] = 0;
    out->outputCount = 0;
    bool ok = false;
    bool outOfMemory = false;
    {
        Compiler compiler(desc, allocator, out);
        ok = compiler.Run();
        outOfMemory = compiler.OutOfMemory();
    }
    if (ok)
    {
        return StatusType::Ok();
    }
    out->words.Clear();
    if (out->diagnostic[0] == 0)
    {
        std::snprintf(out->diagnostic, sizeof(out->diagnostic), "%s",
                      outOfMemory ? "hlsl: out of memory" : "hlsl: compilation failed");
    }
    const bool unsupported = std::strstr(out->diagnostic, "not supported") != nullptr;
    return StatusType::Err(Error{outOfMemory   ? ErrorCode::Internal
                                 : unsupported ? ErrorCode::Unsupported
                                               : ErrorCode::InvalidArgument,
                                 out->diagnostic});
}

}  // namespace cyber::rhi::detail::hlsl
