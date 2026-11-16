// Lexing, SPIR-V module emission, the type table with HLSL layout rules,
// constants, and value conversions for the HLSL compiler (hlsl_spirv_internal.h).

#include "hlsl_spirv_internal.h"

#include <cstdio>
#include <cstdlib>
#include <new>

namespace cyber::rhi::detail::hlsl
{

// ---- Arena ----------------------------------------------------------------------

Arena::~Arena()
{
    Chunk* chunk = m_head;
    while (chunk != nullptr)
    {
        Chunk* next = chunk->next;
        m_allocator->Deallocate(chunk);
        chunk = next;
    }
}

void* Arena::Allocate(usize bytes, usize alignment) noexcept
{
    if (m_failed || bytes == 0)
    {
        return nullptr;
    }
    alignment = alignment < 8 ? 8 : alignment;
    const usize header = (sizeof(Chunk) + 15) & ~static_cast<usize>(15);
    if (m_head != nullptr)
    {
        const usize start = (m_head->used + alignment - 1) & ~(alignment - 1);
        if (start + bytes <= m_head->capacity)
        {
            m_head->used = start + bytes;
            return reinterpret_cast<unsigned char*>(m_head) + start;
        }
    }
    usize capacity = 64 * 1024;
    while (capacity < header + bytes + alignment)
    {
        capacity *= 2;
    }
    void* memory = m_allocator->Allocate(capacity, 16);
    if (memory == nullptr)
    {
        m_failed = true;
        return nullptr;
    }
    Chunk* chunk = static_cast<Chunk*>(memory);
    chunk->next = m_head;
    chunk->capacity = capacity;
    const usize start = (header + alignment - 1) & ~(alignment - 1);
    chunk->used = start + bytes;
    m_head = chunk;
    return reinterpret_cast<unsigned char*>(chunk) + start;
}

// ---- Construction ---------------------------------------------------------------

Compiler::Compiler(const CompileDesc& desc, cyber::memory::Allocator& allocator,
                   CompileOutput* out) noexcept
    : m_desc(desc), m_out(out), m_arena(allocator)
{
    m_tokens.Init(&m_arena);
    m_capabilities.Init(&m_arena);
    m_entryPoints.Init(&m_arena);
    m_executionModes.Init(&m_arena);
    m_debug.Init(&m_arena);
    m_annotations.Init(&m_arena);
    m_globals.Init(&m_arena);
    m_functions.Init(&m_arena);
    m_interface.Init(&m_arena);
    m_types.Init(&m_arena);
    m_funcTypeParams.Init(&m_arena);
    m_constants.Init(&m_arena);
    m_constOperands.Init(&m_arena);
    m_structs.Init(&m_arena);
    m_members.Init(&m_arena);
    m_resources.Init(&m_arena);
    m_bindingVars.Init(&m_arena);
    m_funcs.Init(&m_arena);
    m_params.Init(&m_arena);
    m_symbols.Init(&m_arena);
    m_scopes.Init(&m_arena);
    m_funcStack.Init(&m_arena);
    m_globalScope.variables.Init(&m_arena);
    m_globalScope.body.Init(&m_arena);
    m_globalScope.loops.Init(&m_arena);
    (void)m_types.Push(TypeEntry{});    // TypeId 0 is "none"
    (void)m_structs.Push(StructDef{});  // definition 0 is unused, so indices are 1-based
}

// ---- Diagnostics ------------------------------------------------------------------

bool Compiler::Fail(const Token& at, const char* message) noexcept
{
    if (!m_error)
    {
        m_error = true;
        const char* name = m_desc.sourceName != nullptr ? m_desc.sourceName : "hlsl";
        std::snprintf(m_out->diagnostic, sizeof(m_out->diagnostic), "%s:%u:%u: %s", name,
                      static_cast<unsigned>(at.line), static_cast<unsigned>(at.column), message);
    }
    return false;
}

bool Compiler::FailHere(const char* message) noexcept
{
    return Fail(Peek(), message);
}

// ---- Lexer ------------------------------------------------------------------------

namespace
{

bool IsIdentStart(char c) noexcept
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
bool IsDigit(char c) noexcept
{
    return c >= '0' && c <= '9';
}
bool IsHexDigit(char c) noexcept
{
    return IsDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

constexpr const char* kPunct3[] = {"<<=", ">>="};
constexpr const char* kPunct2[] = {"++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
                                   "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "::"};
constexpr const char kPunct1[] = "(){}[];,.:?+-*/%&|^~!<>=";

}  // namespace

bool Compiler::Lex() noexcept
{
    const char* p = m_desc.source;
    const char* end = p + m_desc.sourceBytes;
    u32 line = 1;
    const char* lineStart = p;
    bool atLineStart = true;
    while (p < end)
    {
        const char c = *p;
        if (c == 0)
        {
            break;
        }
        if (c == '\n')
        {
            ++line;
            ++p;
            lineStart = p;
            atLineStart = true;
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v')
        {
            ++p;
            continue;
        }
        Token token{};
        token.line = line;
        token.column = static_cast<u32>(p - lineStart) + 1;
        if (c == '/' && p + 1 < end && p[1] == '/')
        {
            while (p < end && *p != '\n')
            {
                ++p;
            }
            continue;
        }
        if (c == '/' && p + 1 < end && p[1] == '*')
        {
            p += 2;
            while (p + 1 < end && !(p[0] == '*' && p[1] == '/'))
            {
                if (*p == '\n')
                {
                    ++line;
                    lineStart = p + 1;
                }
                ++p;
            }
            if (p + 1 >= end)
            {
                return Fail(token, "unterminated block comment");
            }
            p += 2;
            continue;
        }
        if (c == '#' && atLineStart)
        {
            const char* directive = p + 1;
            while (directive < end && (*directive == ' ' || *directive == '\t'))
            {
                ++directive;
            }
            const bool skippable =
                (end - directive >= 6 && std::memcmp(directive, "pragma", 6) == 0) ||
                (end - directive >= 4 && std::memcmp(directive, "line", 4) == 0);
            if (!skippable)
            {
                return Fail(token, "preprocessor directives other than #pragma and #line are "
                                   "not supported by the Vulkan HLSL compiler");
            }
            while (p < end && *p != '\n')
            {
                ++p;
            }
            continue;
        }
        atLineStart = false;
        token.text = p;
        if (IsIdentStart(c))
        {
            const char* start = p;
            while (p < end && (IsIdentStart(*p) || IsDigit(*p)))
            {
                ++p;
            }
            token.kind = Tok::Ident;
            token.length = static_cast<u32>(p - start);
        }
        else if (IsDigit(c) || (c == '.' && p + 1 < end && IsDigit(p[1])))
        {
            const char* start = p;
            bool isFloat = false;
            if (c == '0' && p + 1 < end && (p[1] == 'x' || p[1] == 'X'))
            {
                p += 2;
                u64 value = 0;
                const char* digits = p;
                while (p < end && IsHexDigit(*p))
                {
                    const char h = *p;
                    const u64 digit = IsDigit(h) ? static_cast<u64>(h - '0')
                                                 : static_cast<u64>((h | 0x20) - 'a' + 10);
                    value = value * 16 + digit;
                    if (value > 0xFFFFFFFFULL)
                    {
                        return Fail(token, "integer literal does not fit in 32 bits");
                    }
                    ++p;
                }
                if (p == digits)
                {
                    return Fail(token, "malformed hexadecimal literal");
                }
                token.intValue = value;
            }
            else
            {
                while (p < end && IsDigit(*p))
                {
                    ++p;
                }
                if (p < end && *p == '.')
                {
                    isFloat = true;
                    ++p;
                    while (p < end && IsDigit(*p))
                    {
                        ++p;
                    }
                }
                if (p < end && (*p == 'e' || *p == 'E'))
                {
                    const char* exponent = p + 1;
                    if (exponent < end && (*exponent == '+' || *exponent == '-'))
                    {
                        ++exponent;
                    }
                    if (exponent < end && IsDigit(*exponent))
                    {
                        isFloat = true;
                        p = exponent;
                        while (p < end && IsDigit(*p))
                        {
                            ++p;
                        }
                    }
                }
                char buffer[64] = {};
                const usize length = static_cast<usize>(p - start);
                if (length >= sizeof(buffer))
                {
                    return Fail(token, "numeric literal is too long");
                }
                std::memcpy(buffer, start, length);
                if (isFloat)
                {
                    token.floatValue = std::strtod(buffer, nullptr);
                }
                else
                {
                    u64 value = 0;
                    for (usize index = 0; index < length; ++index)
                    {
                        value = value * 10 + static_cast<u64>(buffer[index] - '0');
                        if (value > 0xFFFFFFFFULL)
                        {
                            return Fail(token, "integer literal does not fit in 32 bits");
                        }
                    }
                    token.intValue = value;
                }
            }
            // Suffixes.
            while (p < end && IsIdentStart(*p))
            {
                const char s = static_cast<char>(*p | 0x20);
                if (!isFloat && s == 'u')
                {
                    token.isUnsigned = true;
                }
                else if (s == 'f' || s == 'h')
                {
                    if (!isFloat)
                    {
                        // `1f` is not HLSL, but `1.f` is: only accept after a float.
                        return Fail(token, "malformed numeric literal suffix");
                    }
                }
                else if (s != 'l')
                {
                    return Fail(token, "malformed numeric literal suffix");
                }
                ++p;
            }
            token.kind = isFloat ? Tok::Float : Tok::Int;
            token.length = static_cast<u32>(p - start);
            if (!isFloat && token.intValue > 0x7FFFFFFFULL)
            {
                token.isUnsigned = true;
            }
        }
        else
        {
            token.kind = Tok::Punct;
            bool matched = false;
            for (const char* punct : kPunct3)
            {
                if (end - p >= 3 && std::memcmp(p, punct, 3) == 0)
                {
                    token.length = 3;
                    matched = true;
                    break;
                }
            }
            if (!matched)
            {
                for (const char* punct : kPunct2)
                {
                    if (end - p >= 2 && std::memcmp(p, punct, 2) == 0)
                    {
                        token.length = 2;
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched)
            {
                for (const char* q = kPunct1; *q != 0; ++q)
                {
                    if (*q == c)
                    {
                        token.length = 1;
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched)
            {
                return Fail(token, "unexpected character");
            }
            p += token.length;
        }
        if (!m_tokens.Push(token))
        {
            return Fail(token, "out of memory");
        }
    }
    Token endToken{};
    endToken.kind = Tok::End;
    endToken.line = line;
    endToken.column = static_cast<u32>(p - lineStart) + 1;
    endToken.text = p;
    return m_tokens.Push(endToken) || Fail(endToken, "out of memory");
}

const Token& Compiler::Peek(u32 ahead) const noexcept
{
    const u32 index = m_cursor + ahead;
    const u32 last = m_tokens.Size() == 0 ? 0 : m_tokens.Size() - 1;
    return m_tokens[index < last ? index : last];
}

const Token& Compiler::Next() noexcept
{
    const Token& token = Peek();
    if (token.kind != Tok::End)
    {
        ++m_cursor;
    }
    return token;
}

bool Compiler::IsPunct(const char* text, u32 ahead) const noexcept
{
    const Token& token = Peek(ahead);
    const usize length = std::strlen(text);
    return token.kind == Tok::Punct && token.length == length &&
           std::memcmp(token.text, text, length) == 0;
}

bool Compiler::IsIdent(const char* text, u32 ahead) const noexcept
{
    const Token& token = Peek(ahead);
    const usize length = std::strlen(text);
    return token.kind == Tok::Ident && token.length == length &&
           std::memcmp(token.text, text, length) == 0;
}

bool Compiler::Accept(const char* punct) noexcept
{
    if (IsPunct(punct))
    {
        ++m_cursor;
        return true;
    }
    return false;
}

bool Compiler::Expect(const char* punct, const char* message) noexcept
{
    if (m_error)
    {
        return false;
    }
    if (Accept(punct))
    {
        return true;
    }
    return FailHere(message);
}

Name Compiler::ExpectIdent(const char* message) noexcept
{
    if (m_error)
    {
        return Name{};
    }
    const Token& token = Peek();
    if (token.kind != Tok::Ident)
    {
        (void)FailHere(message);
        return Name{};
    }
    ++m_cursor;
    return Name{token.text, token.length};
}

bool Compiler::SkipBalanced(const char* open, const char* close) noexcept
{
    if (!Expect(open, "expected an opening bracket"))
    {
        return false;
    }
    u32 depth = 1;
    while (depth > 0)
    {
        const Token& token = Peek();
        if (token.kind == Tok::End)
        {
            return Fail(token, "unbalanced brackets");
        }
        if (IsPunct(open))
        {
            ++depth;
        }
        else if (IsPunct(close))
        {
            --depth;
        }
        ++m_cursor;
    }
    return true;
}

// ---- Instruction emission ---------------------------------------------------------

void Compiler::Inst(Vec<u32>& stream, u32 op, const u32* operands, u32 count) noexcept
{
    (void)stream.Push(((count + 1) << 16) | op);
    for (u32 index = 0; index < count; ++index)
    {
        (void)stream.Push(operands[index]);
    }
}

void Compiler::Inst(Vec<u32>& stream, u32 op, std::initializer_list<u32> operands) noexcept
{
    Inst(stream, op, operands.begin(), static_cast<u32>(operands.size()));
}

void Compiler::InstString(Vec<u32>& stream, u32 op, std::initializer_list<u32> head,
                          const char* text, u32 length, const u32* tail, u32 tailCount) noexcept
{
    const u32 stringWords = length / 4 + 1;
    const u32 count = static_cast<u32>(head.size()) + stringWords + tailCount;
    (void)stream.Push(((count + 1) << 16) | op);
    for (const u32 word : head)
    {
        (void)stream.Push(word);
    }
    for (u32 word = 0; word < stringWords; ++word)
    {
        u32 value = 0;
        for (u32 byte = 0; byte < 4; ++byte)
        {
            const u32 index = word * 4 + byte;
            const u32 character = index < length ? static_cast<unsigned char>(text[index]) : 0U;
            value |= character << (byte * 8);
        }
        (void)stream.Push(value);
    }
    for (u32 index = 0; index < tailCount; ++index)
    {
        (void)stream.Push(tail[index]);
    }
}

void Compiler::Capability(u32 capability) noexcept
{
    for (u32 index = 0; index < m_capabilities.Size(); index += 2)
    {
        if (m_capabilities[index + 1] == capability)
        {
            return;
        }
    }
    Inst(m_capabilities, spv::OpCapability, {capability});
}

void Compiler::Decorate(u32 target, u32 decoration, std::initializer_list<u32> extra) noexcept
{
    u32 operands[8] = {target, decoration};
    u32 count = 2;
    for (const u32 word : extra)
    {
        if (count < 8)
        {
            operands[count++] = word;
        }
    }
    Inst(m_annotations, spv::OpDecorate, operands, count);
}

void Compiler::MemberDecorate(u32 type, u32 member, u32 decoration,
                              std::initializer_list<u32> extra) noexcept
{
    u32 operands[8] = {type, member, decoration};
    u32 count = 3;
    for (const u32 word : extra)
    {
        if (count < 8)
        {
            operands[count++] = word;
        }
    }
    Inst(m_annotations, spv::OpMemberDecorate, operands, count);
}

// ---- Types ------------------------------------------------------------------------

TypeId Compiler::Intern(TK kind, u32 a, u32 b, u32 c) noexcept
{
    for (u32 index = 1; index < m_types.Size(); ++index)
    {
        const TypeEntry& entry = m_types[index];
        if (entry.kind == kind && entry.a == a && entry.b == b && entry.c == c)
        {
            return index;
        }
    }
    TypeEntry entry{};
    entry.kind = kind;
    entry.a = a;
    entry.b = b;
    entry.c = c;
    if (!m_types.Push(entry))
    {
        return 0;
    }
    return m_types.Size() - 1;
}

TypeId Compiler::VectorOf(TypeId scalar, u32 count) noexcept
{
    return count <= 1 ? scalar : Intern(TK::Vector, scalar, count);
}

TypeId Compiler::MatrixOf(TypeId scalar, u32 rows, u32 columns) noexcept
{
    // An HLSL floatRxC: R SPIR-V columns, each a float vector of C components.
    return Intern(TK::Matrix, VectorOf(scalar, columns), rows);
}

TypeId Compiler::PointerTo(u32 storage, TypeId pointee) noexcept
{
    return Intern(TK::Pointer, storage, pointee);
}

TypeId Compiler::ArrayOf(TypeId element, u32 length, u32 stride) noexcept
{
    return Intern(TK::Array, element, length, stride);
}

TypeId Compiler::RuntimeArrayOf(TypeId element, u32 stride) noexcept
{
    return Intern(TK::RtArray, element, 0, stride);
}

TypeId Compiler::FunctionType(TypeId result, const TypeId* params, u32 count) noexcept
{
    for (u32 index = 1; index < m_types.Size(); ++index)
    {
        const TypeEntry& entry = m_types[index];
        if (entry.kind != TK::Function || entry.a != result || entry.c != count)
        {
            continue;
        }
        bool same = true;
        for (u32 param = 0; param < count && same; ++param)
        {
            same = m_funcTypeParams[entry.b + param] == params[param];
        }
        if (same)
        {
            return index;
        }
    }
    const u32 first = m_funcTypeParams.Size();
    for (u32 param = 0; param < count; ++param)
    {
        (void)m_funcTypeParams.Push(params[param]);
    }
    TypeEntry entry{};
    entry.kind = TK::Function;
    entry.a = result;
    entry.b = first;
    entry.c = count;
    if (!m_types.Push(entry))
    {
        return 0;
    }
    return m_types.Size() - 1;
}

u32 Compiler::SpirvType(TypeId type) noexcept
{
    if (type == 0 || type >= m_types.Size())
    {
        return 0;
    }
    if (m_types[type].id != 0)
    {
        return m_types[type].id;
    }
    const TypeEntry entry = m_types[type];
    u32 id = 0;
    switch (entry.kind)
    {
    case TK::None:
        return 0;
    case TK::Void:
        id = NewId();
        Inst(m_globals, spv::OpTypeVoid, {id});
        break;
    case TK::Bool:
        id = NewId();
        Inst(m_globals, spv::OpTypeBool, {id});
        break;
    case TK::Int:
        id = NewId();
        Inst(m_globals, spv::OpTypeInt, {id, 32, 1});
        break;
    case TK::Uint:
        id = NewId();
        Inst(m_globals, spv::OpTypeInt, {id, 32, 0});
        break;
    case TK::Float:
        id = NewId();
        Inst(m_globals, spv::OpTypeFloat, {id, 32});
        break;
    case TK::Vector:
    {
        const u32 element = SpirvType(entry.a);
        id = NewId();
        Inst(m_globals, spv::OpTypeVector, {id, element, entry.b});
        break;
    }
    case TK::Matrix:
    {
        const u32 column = SpirvType(entry.a);
        id = NewId();
        Inst(m_globals, spv::OpTypeMatrix, {id, column, entry.b});
        break;
    }
    case TK::Array:
    {
        const u32 element = SpirvType(entry.a);
        const u32 length = ConstUint(entry.b);
        id = NewId();
        Inst(m_globals, spv::OpTypeArray, {id, element, length});
        if (entry.c != 0)
        {
            Decorate(id, spv::DecArrayStride, {entry.c});
        }
        break;
    }
    case TK::RtArray:
    {
        const u32 element = SpirvType(entry.a);
        id = NewId();
        Inst(m_globals, spv::OpTypeRuntimeArray, {id, element});
        if (entry.c != 0)
        {
            Decorate(id, spv::DecArrayStride, {entry.c});
        }
        break;
    }
    case TK::Struct:
    {
        const StructDef& def = m_structs[entry.a];
        u32 memberIds[64] = {};
        const Layout layout = static_cast<Layout>(entry.b);
        if (def.memberCount > 64)
        {
            (void)FailHere("structures are limited to 64 members");
            return 0;
        }
        for (u32 member = 0; member < def.memberCount; ++member)
        {
            const TypeId logical = m_members[def.firstMember + member].type;
            memberIds[member] = SpirvType(LaidOut(logical, layout, false));
        }
        id = NewId();
        u32 operands[65] = {id};
        for (u32 member = 0; member < def.memberCount; ++member)
        {
            operands[member + 1] = memberIds[member];
        }
        Inst(m_globals, spv::OpTypeStruct, operands, def.memberCount + 1);
        if (entry.c != 0)
        {
            Decorate(id, spv::DecBlock);
        }
        if (layout != LayoutLogical)
        {
            u32 offset = 0;
            for (u32 member = 0; member < def.memberCount; ++member)
            {
                const StructMember& info = m_members[def.firstMember + member];
                u32 size = 0;
                u32 align = 0;
                (void)SizeAndAlign(info.type, layout, &size, &align);
                if (layout == LayoutCBuffer)
                {
                    if (info.hasPackOffset)
                    {
                        offset = info.packOffsetBytes;
                    }
                    else if (align == 16 || (offset % 16) + size > 16)
                    {
                        offset = (offset + 15) & ~15U;
                    }
                }
                MemberDecorate(id, member, spv::DecOffset, {offset});
                TypeId inner = info.type;
                while (T(inner).kind == TK::Array || T(inner).kind == TK::RtArray)
                {
                    inner = T(inner).a;
                }
                if (T(inner).kind == TK::Matrix)
                {
                    const u32 rows = T(inner).b;  // HLSL rows = SPIR-V columns
                    const u32 stride = layout == LayoutCBuffer ? 16 : rows * 4;
                    MemberDecorate(id, member, spv::DecRowMajor);
                    MemberDecorate(id, member, spv::DecMatrixStride, {stride});
                }
                offset += size;
            }
        }
        break;
    }
    case TK::Image:
    {
        const u32 sampled = SpirvType(entry.a);
        id = NewId();
        const u32 arrayed = entry.c & 1;
        const u32 ms = (entry.c >> 1) & 1;
        const u32 sampledMode = (entry.c >> 2) & 3;
        const u32 depth = (entry.c >> 4) & 3;
        Inst(m_globals, spv::OpTypeImage,
             {id, sampled, entry.b, depth, arrayed, ms, sampledMode, 0});
        break;
    }
    case TK::Sampler:
        id = NewId();
        Inst(m_globals, spv::OpTypeSampler, {id});
        break;
    case TK::SampledImage:
    {
        const u32 image = SpirvType(entry.a);
        id = NewId();
        Inst(m_globals, spv::OpTypeSampledImage, {id, image});
        break;
    }
    case TK::Pointer:
    {
        const u32 pointee = SpirvType(entry.b);
        id = NewId();
        Inst(m_globals, spv::OpTypePointer, {id, entry.a, pointee});
        break;
    }
    case TK::Function:
    {
        u32 operands[34] = {};
        const u32 result = SpirvType(entry.a);
        if (entry.c > 32)
        {
            (void)FailHere("functions are limited to 32 parameters");
            return 0;
        }
        for (u32 param = 0; param < entry.c; ++param)
        {
            operands[param + 2] = SpirvType(m_funcTypeParams[entry.b + param]);
        }
        id = NewId();
        operands[0] = id;
        operands[1] = result;
        Inst(m_globals, spv::OpTypeFunction, operands, entry.c + 2);
        break;
    }
    }
    m_types[type].id = id;
    return id;
}

bool Compiler::IsScalar(TypeId type) const noexcept
{
    const TK kind = T(type).kind;
    return kind == TK::Bool || kind == TK::Int || kind == TK::Uint || kind == TK::Float;
}

bool Compiler::IsNumericScalarOrVector(TypeId type) const noexcept
{
    return IsScalar(type) || T(type).kind == TK::Vector;
}

TypeId Compiler::ScalarOf(TypeId type) const noexcept
{
    const TypeEntry& entry = T(type);
    if (entry.kind == TK::Vector)
    {
        return entry.a;
    }
    if (entry.kind == TK::Matrix)
    {
        return T(entry.a).a;
    }
    return IsScalar(type) ? type : 0;
}

u32 Compiler::ComponentCount(TypeId type) const noexcept
{
    const TypeEntry& entry = T(type);
    if (entry.kind == TK::Vector)
    {
        return entry.b;
    }
    if (entry.kind == TK::Matrix)
    {
        return entry.b * T(entry.a).b;
    }
    return 1;
}

TypeId Compiler::WithScalar(TypeId shape, TypeId scalar) noexcept
{
    const TypeEntry entry = T(shape);
    if (entry.kind == TK::Vector)
    {
        return VectorOf(scalar, entry.b);
    }
    if (entry.kind == TK::Matrix)
    {
        return Intern(TK::Matrix, VectorOf(scalar, T(entry.a).b), entry.b);
    }
    return scalar;
}

TypeId Compiler::WithCount(TypeId scalar, u32 count) noexcept
{
    return VectorOf(scalar, count);
}

bool Compiler::IsFloatish(TypeId type) const noexcept
{
    return T(ScalarOf(type)).kind == TK::Float;
}

bool Compiler::IsSigned(TypeId type) const noexcept
{
    return T(ScalarOf(type)).kind == TK::Int;
}

bool Compiler::IsBoolish(TypeId type) const noexcept
{
    return T(ScalarOf(type)).kind == TK::Bool;
}

bool Compiler::IsInteger(TypeId type) const noexcept
{
    const TK kind = T(ScalarOf(type)).kind;
    return kind == TK::Int || kind == TK::Uint;
}

TypeId Compiler::StructType(u32 def, Layout layout, bool block) noexcept
{
    return Intern(TK::Struct, def, layout, block ? 1U : 0U);
}

TypeId Compiler::LogicalOf(TypeId type) noexcept
{
    const TypeEntry entry = T(type);
    switch (entry.kind)
    {
    case TK::Array:
        return ArrayOf(LogicalOf(entry.a), entry.b, 0);
    case TK::RtArray:
        return RuntimeArrayOf(LogicalOf(entry.a), 0);
    case TK::Struct:
        return StructType(entry.a, LayoutLogical, false);
    default:
        return type;
    }
}

bool Compiler::SizeAndAlign(TypeId logical, Layout layout, u32* size, u32* align) noexcept
{
    const TypeEntry entry = T(logical);
    switch (entry.kind)
    {
    case TK::Bool:
    case TK::Int:
    case TK::Uint:
    case TK::Float:
        *size = 4;
        *align = 4;
        return true;
    case TK::Vector:
        *size = 4 * entry.b;
        *align = 4;
        return true;
    case TK::Matrix:
    {
        const u32 rows = entry.b;
        const u32 columns = T(entry.a).b;
        if (layout == LayoutCBuffer)
        {
            // Column-major: one 16-byte register per HLSL column of `rows` floats.
            *size = 16 * (columns - 1) + 4 * rows;
            *align = 16;
        }
        else
        {
            *size = 4 * rows * columns;
            *align = 4;
        }
        return true;
    }
    case TK::Array:
    {
        u32 elementSize = 0;
        u32 elementAlign = 0;
        if (!SizeAndAlign(entry.a, layout, &elementSize, &elementAlign))
        {
            return false;
        }
        if (layout == LayoutCBuffer)
        {
            const u32 stride = (elementSize + 15) & ~15U;
            *size = entry.b == 0 ? 0 : stride * (entry.b - 1) + elementSize;
            *align = 16;
        }
        else
        {
            *size = elementSize * entry.b;
            *align = 4;
        }
        return true;
    }
    case TK::RtArray:
        *size = 0;
        *align = layout == LayoutCBuffer ? 16 : 4;
        return true;
    case TK::Struct:
    {
        const StructDef& def = m_structs[entry.a];
        u32 offset = 0;
        for (u32 member = 0; member < def.memberCount; ++member)
        {
            const StructMember& info = m_members[def.firstMember + member];
            u32 memberSize = 0;
            u32 memberAlign = 0;
            if (!SizeAndAlign(info.type, layout, &memberSize, &memberAlign))
            {
                return false;
            }
            if (layout == LayoutCBuffer)
            {
                if (info.hasPackOffset)
                {
                    offset = info.packOffsetBytes;
                }
                else if (memberAlign == 16 || (offset % 16) + memberSize > 16)
                {
                    offset = (offset + 15) & ~15U;
                }
            }
            offset += memberSize;
        }
        *size = offset;
        *align = layout == LayoutCBuffer ? 16 : 4;
        return true;
    }
    default:
        *size = 0;
        *align = 4;
        return false;
    }
}

TypeId Compiler::LaidOut(TypeId logical, Layout layout, bool block) noexcept
{
    if (layout == LayoutLogical)
    {
        return logical;
    }
    const TypeEntry entry = T(logical);
    switch (entry.kind)
    {
    case TK::Array:
    {
        u32 size = 0;
        u32 align = 0;
        (void)SizeAndAlign(entry.a, layout, &size, &align);
        const u32 stride = layout == LayoutCBuffer ? ((size + 15) & ~15U) : size;
        return ArrayOf(LaidOut(entry.a, layout, false), entry.b, stride == 0 ? 4 : stride);
    }
    case TK::RtArray:
    {
        u32 size = 0;
        u32 align = 0;
        (void)SizeAndAlign(entry.a, layout, &size, &align);
        const u32 stride = layout == LayoutCBuffer ? ((size + 15) & ~15U) : size;
        return RuntimeArrayOf(LaidOut(entry.a, layout, false), stride == 0 ? 4 : stride);
    }
    case TK::Struct:
        return StructType(entry.a, layout, block);
    default:
        return logical;
    }
}

// ---- Constants --------------------------------------------------------------------

u32 Compiler::ConstantBits(TypeId scalar, u32 bits) noexcept
{
    u32 op = spv::OpConstant;
    if (T(scalar).kind == TK::Bool)
    {
        op = bits != 0 ? spv::OpConstantTrue : spv::OpConstantFalse;
    }
    for (u32 index = 0; index < m_constants.Size(); ++index)
    {
        const ConstKey& key = m_constants[index];
        if (key.op == op && key.type == scalar &&
            (op != spv::OpConstant || (key.count == 1 && m_constOperands[key.first] == bits)))
        {
            return key.id;
        }
    }
    const u32 typeId = SpirvType(scalar);
    const u32 id = NewId();
    if (op == spv::OpConstant)
    {
        Inst(m_globals, op, {typeId, id, bits});
    }
    else
    {
        Inst(m_globals, op, {typeId, id});
    }
    ConstKey key{op, scalar, m_constOperands.Size(), 1, id};
    (void)m_constOperands.Push(bits);
    (void)m_constants.Push(key);
    return id;
}

u32 Compiler::ConstFloat(float value) noexcept
{
    return ConstantBits(FloatType(), FloatBits(value));
}

u32 Compiler::ConstantComposite(TypeId type, const u32* ids, u32 count) noexcept
{
    for (u32 index = 0; index < m_constants.Size(); ++index)
    {
        const ConstKey& key = m_constants[index];
        if (key.op != spv::OpConstantComposite || key.type != type || key.count != count)
        {
            continue;
        }
        bool same = true;
        for (u32 component = 0; component < count && same; ++component)
        {
            same = m_constOperands[key.first + component] == ids[component];
        }
        if (same)
        {
            return key.id;
        }
    }
    const u32 typeId = SpirvType(type);
    const u32 id = NewId();
    (void)m_globals.Push(((count + 3) << 16) | spv::OpConstantComposite);
    (void)m_globals.Push(typeId);
    (void)m_globals.Push(id);
    ConstKey key{spv::OpConstantComposite, type, m_constOperands.Size(), count, id};
    for (u32 component = 0; component < count; ++component)
    {
        (void)m_globals.Push(ids[component]);
        (void)m_constOperands.Push(ids[component]);
    }
    (void)m_constants.Push(key);
    return id;
}

u32 Compiler::ConstantNull(TypeId type) noexcept
{
    for (u32 index = 0; index < m_constants.Size(); ++index)
    {
        const ConstKey& key = m_constants[index];
        if (key.op == spv::OpConstantNull && key.type == type)
        {
            return key.id;
        }
    }
    const u32 typeId = SpirvType(type);
    const u32 id = NewId();
    Inst(m_globals, spv::OpConstantNull, {typeId, id});
    (void)m_constants.Push(ConstKey{spv::OpConstantNull, type, 0, 0, id});
    return id;
}

Val Compiler::ConstVal(TypeId scalar, u32 bits) noexcept
{
    Val value{};
    value.kind = VK::R;
    value.type = scalar;
    value.isConst = true;
    value.constBits = T(scalar).kind == TK::Bool ? (bits != 0 ? 1U : 0U) : bits;
    value.id = ConstantBits(scalar, value.constBits);
    return value;
}

Val Compiler::Splat(Val scalar, TypeId vectorType) noexcept
{
    const u32 count = ComponentCount(vectorType);
    if (count <= 1)
    {
        return scalar;
    }
    u32 ids[4] = {scalar.id, scalar.id, scalar.id, scalar.id};
    Val result{};
    result.kind = VK::R;
    result.type = vectorType;
    if (scalar.isConst)
    {
        result.id = ConstantComposite(vectorType, ids, count);
    }
    else
    {
        result.id = OpN(spv::OpCompositeConstruct, vectorType, ids, count);
    }
    return result;
}

// ---- Function-body emission -------------------------------------------------------

Vec<u32>& Compiler::Body() noexcept
{
    if (m_funcStack.Size() == 0)
    {
        return m_globalScope.body;
    }
    return m_funcStack.Back()->body;
}

void Compiler::Open() noexcept
{
    if (m_funcStack.Size() == 0)
    {
        return;
    }
    FuncState& state = *m_funcStack.Back();
    if (!state.blockOpen)
    {
        Inst(state.body, spv::OpLabel, {NewId()});
        state.blockOpen = true;
    }
}

u32 Compiler::OpN(u32 op, TypeId result, const u32* operands, u32 count) noexcept
{
    if (m_atGlobalScope)
    {
        (void)FailHere("a global initializer must be a constant expression");
        return 0;
    }
    Open();
    const u32 id = NewId();
    Vec<u32>& body = Body();
    (void)body.Push(((count + 3) << 16) | op);
    (void)body.Push(SpirvType(result));
    (void)body.Push(id);
    for (u32 index = 0; index < count; ++index)
    {
        (void)body.Push(operands[index]);
    }
    return id;
}

u32 Compiler::Op(u32 op, TypeId result, std::initializer_list<u32> operands) noexcept
{
    return OpN(op, result, operands.begin(), static_cast<u32>(operands.size()));
}

void Compiler::OpVoid(u32 op, std::initializer_list<u32> operands) noexcept
{
    if (m_atGlobalScope)
    {
        (void)FailHere("a global initializer must be a constant expression");
        return;
    }
    Open();
    Inst(Body(), op, operands);
}

u32 Compiler::Ext(u32 instruction, TypeId result, std::initializer_list<u32> operands) noexcept
{
    u32 words[8] = {m_glsl, instruction};
    u32 count = 2;
    for (const u32 word : operands)
    {
        if (count < 8)
        {
            words[count++] = word;
        }
    }
    return OpN(spv::OpExtInst, result, words, count);
}

void Compiler::Label(u32 id) noexcept
{
    if (m_funcStack.Size() == 0)
    {
        return;
    }
    FuncState& state = *m_funcStack.Back();
    if (state.blockOpen)
    {
        // Fall through into the new block.
        Inst(state.body, spv::OpBranch, {id});
    }
    Inst(state.body, spv::OpLabel, {id});
    state.blockOpen = true;
}

void Compiler::Branch(u32 target) noexcept
{
    if (m_funcStack.Size() == 0)
    {
        return;
    }
    FuncState& state = *m_funcStack.Back();
    if (state.blockOpen)
    {
        Inst(state.body, spv::OpBranch, {target});
        state.blockOpen = false;
    }
}

void Compiler::Terminate(u32 op, std::initializer_list<u32> operands) noexcept
{
    if (m_funcStack.Size() == 0)
    {
        return;
    }
    Open();
    FuncState& state = *m_funcStack.Back();
    Inst(state.body, op, operands);
    state.blockOpen = false;
}

u32 Compiler::NewLocal(TypeId type) noexcept
{
    if (m_funcStack.Size() == 0)
    {
        (void)FailHere("a local variable outside a function");
        return 0;
    }
    const u32 id = NewId();
    Inst(m_funcStack.Back()->variables, spv::OpVariable,
         {SpirvType(PointerTo(spv::Function, type)), id, spv::Function});
    return id;
}

// ---- Loads, stores and conversions ------------------------------------------------

Val Compiler::Load(const Val& value) noexcept
{
    Val result{};
    result.kind = VK::R;
    switch (value.kind)
    {
    case VK::None:
        return Val{};
    case VK::R:
        return value;
    case VK::Ptr:
    {
        if (T(value.type).kind == TK::RtArray)
        {
            (void)FailHere("a runtime-sized array cannot be read as a value");
            return Val{};
        }
        result.type = value.type;
        result.id = Op(spv::OpLoad, value.type, {value.id});
        if (value.nonUniform)
        {
            Decorate(result.id, spv::DecNonUniform);
        }
        const TypeId logical = LogicalOf(value.type);
        if (logical != value.type)
        {
            result = ConvertLayout(result, logical);
        }
        return result;
    }
    case VK::Swz:
    {
        Val vector{};
        vector.kind = VK::R;
        vector.type = value.type;
        vector.id = Op(spv::OpLoad, value.type, {value.id});
        const TypeId scalar = ScalarOf(value.type);
        if (value.swizzleCount == 1)
        {
            result.type = scalar;
            result.id = Op(spv::OpCompositeExtract, scalar, {vector.id, value.swizzle[0]});
        }
        else
        {
            result.type = VectorOf(scalar, value.swizzleCount);
            u32 operands[6] = {vector.id, vector.id};
            for (u32 index = 0; index < value.swizzleCount; ++index)
            {
                operands[2 + index] = value.swizzle[index];
            }
            result.id = OpN(spv::OpVectorShuffle, result.type, operands, 2U + value.swizzleCount);
        }
        return result;
    }
    case VK::Texel:
    {
        if (!m_desc.features.storageImageReadWithoutFormat)
        {
            (void)FailHere("reading a storage image needs shaderStorageImageReadWithoutFormat, "
                           "which this device does not support");
            return Val{};
        }
        Capability(spv::CapStorageImageReadWithoutFormat);
        const TypeId scalar = ScalarOf(value.type);
        const TypeId texel = VectorOf(scalar, 4);
        const u32 read = Op(spv::OpImageRead, texel, {value.id, value.aux});
        const u32 count = ComponentCount(value.type);
        result.type = value.type;
        if (count == 4)
        {
            result.id = read;
        }
        else if (count == 1)
        {
            result.id = Op(spv::OpCompositeExtract, scalar, {read, 0});
        }
        else
        {
            u32 operands[6] = {read, read, 0, 1, 2, 3};
            result.id = OpN(spv::OpVectorShuffle, value.type, operands, 2 + count);
        }
        return result;
    }
    case VK::Resource:
        (void)FailHere("a resource cannot be used as a value here");
        return Val{};
    }
    return Val{};
}

Val Compiler::Rvalue(const Val& value) noexcept
{
    return Load(value);
}

bool Compiler::Store(const Val& target, Val value) noexcept
{
    if (m_error)
    {
        return false;
    }
    if (target.readOnly)
    {
        return FailHere("cannot assign to a read-only value");
    }
    switch (target.kind)
    {
    case VK::Ptr:
    {
        const TypeId logical = LogicalOf(target.type);
        Val converted = Convert(value, logical);
        if (converted.kind == VK::None)
        {
            return false;
        }
        if (logical != target.type)
        {
            converted = ConvertLayout(converted, target.type);
        }
        OpVoid(spv::OpStore, {target.id, converted.id});
        return !m_error;
    }
    case VK::Swz:
    {
        const TypeId scalar = ScalarOf(target.type);
        if (target.swizzleCount == 1)
        {
            const Val converted = Convert(value, scalar);
            if (converted.kind == VK::None)
            {
                return false;
            }
            const u32 pointer = Op(spv::OpAccessChain, PointerTo(target.storage, scalar),
                                   {target.id, ConstUint(target.swizzle[0])});
            OpVoid(spv::OpStore, {pointer, converted.id});
            return !m_error;
        }
        const TypeId sourceType = VectorOf(scalar, target.swizzleCount);
        const Val converted = Convert(value, sourceType);
        if (converted.kind == VK::None)
        {
            return false;
        }
        const u32 old = Op(spv::OpLoad, target.type, {target.id});
        const u32 width = ComponentCount(target.type);
        u32 operands[6] = {old, converted.id};
        for (u32 component = 0; component < width; ++component)
        {
            u32 select = component;
            for (u32 index = 0; index < target.swizzleCount; ++index)
            {
                if (target.swizzle[index] == component)
                {
                    select = width + index;
                }
            }
            operands[2 + component] = select;
        }
        const u32 merged = OpN(spv::OpVectorShuffle, target.type, operands, 2 + width);
        OpVoid(spv::OpStore, {target.id, merged});
        return !m_error;
    }
    case VK::Texel:
    {
        if (!m_desc.features.storageImageWriteWithoutFormat)
        {
            return FailHere("writing a storage image needs shaderStorageImageWriteWithoutFormat, "
                            "which this device does not support");
        }
        Capability(spv::CapStorageImageWriteWithoutFormat);
        const TypeId scalar = ScalarOf(target.type);
        const Val converted = Convert(value, target.type);
        if (converted.kind == VK::None)
        {
            return false;
        }
        const u32 count = ComponentCount(target.type);
        u32 texel = converted.id;
        if (count != 4)
        {
            const u32 zero = ConstantBits(scalar, 0);
            u32 parts[4] = {zero, zero, zero, zero};
            if (count == 1)
            {
                parts[0] = converted.id;
            }
            else
            {
                for (u32 component = 0; component < count; ++component)
                {
                    parts[component] =
                        Op(spv::OpCompositeExtract, scalar, {converted.id, component});
                }
            }
            texel = OpN(spv::OpCompositeConstruct, VectorOf(scalar, 4), parts, 4);
        }
        OpVoid(spv::OpImageWrite, {target.id, target.aux, texel});
        return !m_error;
    }
    default:
        return FailHere("the left side of an assignment is not assignable");
    }
}

namespace
{

// Folds a scalar conversion of a constant. Returns false when it cannot.
bool FoldConvert(TK from, TK to, u32 bits, u32* out) noexcept
{
    switch (to)
    {
    case TK::Float:
        if (from == TK::Int)
        {
            *out = FloatBits(static_cast<float>(static_cast<int>(bits)));
        }
        else if (from == TK::Uint)
        {
            *out = FloatBits(static_cast<float>(bits));
        }
        else if (from == TK::Bool)
        {
            *out = FloatBits(bits != 0 ? 1.0F : 0.0F);
        }
        else
        {
            *out = bits;
        }
        return true;
    case TK::Int:
    case TK::Uint:
        if (from == TK::Float)
        {
            const float value = BitsFloat(bits);
            if (!(value == value))
            {
                return false;
            }
            if (to == TK::Int)
            {
                if (value >= 2147483648.0F || value < -2147483648.0F)
                {
                    return false;
                }
                *out = static_cast<u32>(static_cast<int>(value));
            }
            else
            {
                if (value >= 4294967296.0F || value <= -1.0F)
                {
                    return false;
                }
                *out = static_cast<u32>(value);
            }
        }
        else if (from == TK::Bool)
        {
            *out = bits != 0 ? 1U : 0U;
        }
        else
        {
            *out = bits;
        }
        return true;
    case TK::Bool:
        if (from == TK::Float)
        {
            *out = BitsFloat(bits) != 0.0F ? 1U : 0U;
        }
        else
        {
            *out = bits != 0 ? 1U : 0U;
        }
        return true;
    default:
        return false;
    }
}

}  // namespace

Val Compiler::Convert(Val value, TypeId target) noexcept
{
    if (m_error)
    {
        return Val{};
    }
    Val source = Rvalue(value);
    if (source.kind == VK::None)
    {
        return Val{};
    }
    if (source.type == target)
    {
        return source;
    }
    const TypeEntry from = T(source.type);
    const TypeEntry to = T(target);
    if ((from.kind == TK::Struct || from.kind == TK::Array) && from.kind == to.kind &&
        LogicalOf(source.type) == LogicalOf(target))
    {
        return ConvertLayout(source, target);
    }
    if (from.kind == TK::Matrix || to.kind == TK::Matrix)
    {
        if (from.kind == TK::Matrix && to.kind == TK::Matrix && from.b == to.b &&
            T(from.a).b == T(to.a).b)
        {
            return source;  // only float matrices exist
        }
        (void)FailHere("this matrix conversion is not supported by the Vulkan HLSL compiler");
        return Val{};
    }
    if (!IsNumericScalarOrVector(source.type) || !IsNumericScalarOrVector(target))
    {
        (void)FailHere("cannot convert between these types");
        return Val{};
    }
    const u32 fromCount = ComponentCount(source.type);
    const u32 toCount = ComponentCount(target);
    const TypeId targetScalar = ScalarOf(target);

    // Shape first: truncate a vector, or take component 0 for a scalar target.
    if (fromCount > toCount)
    {
        const TypeId truncated = VectorOf(ScalarOf(source.type), toCount);
        Val narrowed{};
        narrowed.kind = VK::R;
        narrowed.type = truncated;
        if (toCount == 1)
        {
            narrowed.id = Op(spv::OpCompositeExtract, truncated, {source.id, 0});
        }
        else
        {
            u32 operands[6] = {source.id, source.id, 0, 1, 2, 3};
            narrowed.id = OpN(spv::OpVectorShuffle, truncated, operands, 2 + toCount);
        }
        source = narrowed;
    }
    else if (fromCount < toCount && fromCount != 1)
    {
        (void)FailHere("cannot convert a vector to a wider vector implicitly");
        return Val{};
    }

    // Then the component kind, on the source shape.
    const TK fromKind = T(ScalarOf(source.type)).kind;
    const TK toKind = T(targetScalar).kind;
    if (fromKind != toKind)
    {
        const TypeId converted = WithScalar(source.type, targetScalar);
        if (source.isConst && IsScalar(source.type))
        {
            u32 bits = 0;
            if (FoldConvert(fromKind, toKind, source.constBits, &bits))
            {
                source = ConstVal(targetScalar, bits);
                return fromCount < toCount ? Splat(source, target) : source;
            }
        }
        Val result{};
        result.kind = VK::R;
        result.type = converted;
        if (fromKind == TK::Bool)
        {
            u32 one = 0;
            u32 zero = 0;
            if (toKind == TK::Float)
            {
                one = ConstFloat(1.0F);
                zero = ConstFloat(0.0F);
            }
            else
            {
                one = ConstantBits(targetScalar, 1);
                zero = ConstantBits(targetScalar, 0);
            }
            const u32 count = ComponentCount(converted);
            if (count > 1)
            {
                u32 ones[4] = {one, one, one, one};
                u32 zeros[4] = {zero, zero, zero, zero};
                one = ConstantComposite(converted, ones, count);
                zero = ConstantComposite(converted, zeros, count);
            }
            result.id = Op(spv::OpSelect, converted, {source.id, one, zero});
        }
        else if (toKind == TK::Bool)
        {
            u32 zero =
                fromKind == TK::Float ? ConstFloat(0.0F) : ConstantBits(ScalarOf(source.type), 0);
            const u32 count = ComponentCount(source.type);
            if (count > 1)
            {
                u32 zeros[4] = {zero, zero, zero, zero};
                zero = ConstantComposite(source.type, zeros, count);
            }
            result.id = Op(fromKind == TK::Float ? spv::OpFUnordNotEqual : spv::OpINotEqual,
                           converted, {source.id, zero});
        }
        else if (toKind == TK::Float)
        {
            result.id = Op(fromKind == TK::Int ? spv::OpConvertSToF : spv::OpConvertUToF, converted,
                           {source.id});
        }
        else if (fromKind == TK::Float)
        {
            // D3D's ftoi/ftou saturate and turn NaN into 0; SPIR-V leaves both
            // undefined, and an index built from an out-of-range float then
            // points anywhere. Clamp first so the conversion is defined.
            const u32 count = ComponentCount(source.type);
            const float low = toKind == TK::Int ? -2147483648.0F : 0.0F;
            const float high = toKind == TK::Int ? 2147483520.0F : 4294967040.0F;
            u32 lowId = ConstFloat(low);
            u32 highId = ConstFloat(high);
            u32 zeroId = ConstFloat(0.0F);
            if (count > 1)
            {
                u32 lows[4] = {lowId, lowId, lowId, lowId};
                u32 highs[4] = {highId, highId, highId, highId};
                u32 zeros[4] = {zeroId, zeroId, zeroId, zeroId};
                lowId = ConstantComposite(source.type, lows, count);
                highId = ConstantComposite(source.type, highs, count);
                zeroId = ConstantComposite(source.type, zeros, count);
            }
            const TypeId boolShape = WithScalar(source.type, BoolType());
            const u32 isNan = Op(spv::OpIsNan, boolShape, {source.id});
            const u32 finite = Op(spv::OpSelect, source.type, {isNan, zeroId, source.id});
            const u32 clamped = Ext(spv::GlslFClamp, source.type, {finite, lowId, highId});
            result.id = Op(toKind == TK::Int ? spv::OpConvertFToS : spv::OpConvertFToU, converted,
                           {clamped});
        }
        else
        {
            result.id = Op(spv::OpBitcast, converted, {source.id});
        }
        source = result;
    }
    if (ComponentCount(source.type) < toCount)
    {
        source = Splat(source, target);
    }
    return source;
}

Val Compiler::ConvertLayout(Val value, TypeId target) noexcept
{
    if (m_error || value.kind != VK::R)
    {
        return value;
    }
    if (value.type == target)
    {
        return value;
    }
    const TypeEntry from = T(value.type);
    const TypeEntry to = T(target);
    Val result{};
    result.kind = VK::R;
    result.type = target;
    if (from.kind == TK::Struct && to.kind == TK::Struct && from.a == to.a)
    {
        const StructDef& def = m_structs[from.a];
        u32 parts[64] = {};
        for (u32 member = 0; member < def.memberCount && member < 64; ++member)
        {
            const TypeId logical = m_members[def.firstMember + member].type;
            const TypeId fromMember = LaidOut(logical, static_cast<Layout>(from.b), false);
            const TypeId toMember = LaidOut(logical, static_cast<Layout>(to.b), false);
            Val part{};
            part.kind = VK::R;
            part.type = fromMember;
            part.id = Op(spv::OpCompositeExtract, fromMember, {value.id, member});
            parts[member] = ConvertLayout(part, toMember).id;
        }
        result.id = OpN(spv::OpCompositeConstruct, target, parts, def.memberCount);
        return result;
    }
    if (from.kind == TK::Array && to.kind == TK::Array && from.b == to.b)
    {
        if (from.b > 256)
        {
            (void)FailHere("arrays over 256 elements cannot be copied between buffer and "
                           "local memory as a whole; copy them element by element");
            return Val{};
        }
        u32 parts[256] = {};
        for (u32 element = 0; element < from.b; ++element)
        {
            Val part{};
            part.kind = VK::R;
            part.type = from.a;
            part.id = Op(spv::OpCompositeExtract, from.a, {value.id, element});
            parts[element] = ConvertLayout(part, to.a).id;
        }
        result.id = OpN(spv::OpCompositeConstruct, target, parts, from.b);
        return result;
    }
    (void)FailHere("internal error: incompatible layout conversion");
    return Val{};
}

Val Compiler::ToBool(Val value) noexcept
{
    const Val source = Rvalue(value);
    if (source.kind == VK::None)
    {
        return Val{};
    }
    return Convert(source, WithScalar(source.type, BoolType()));
}

}  // namespace cyber::rhi::detail::hlsl
