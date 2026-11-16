// Expressions, function calls, intrinsics, resources and atomics for the HLSL
// compiler (hlsl_spirv_internal.h).

#include "hlsl_spirv_internal.h"

namespace cyber::rhi::detail::hlsl
{

namespace
{

// Rank of a component kind in HLSL's usual arithmetic conversions.
int Rank(TK kind) noexcept
{
    switch (kind)
    {
    case TK::Bool:
        return 0;
    case TK::Int:
        return 1;
    case TK::Uint:
        return 2;
    case TK::Float:
        return 3;
    default:
        return -1;
    }
}

int Precedence(const Token& token) noexcept
{
    if (token.kind != Tok::Punct)
    {
        return -1;
    }
    const auto is = [&](const char* text) noexcept
    {
        const usize length = std::strlen(text);
        return token.length == length && std::memcmp(token.text, text, length) == 0;
    };
    if (is("||"))
        return 1;
    if (is("&&"))
        return 2;
    if (is("|"))
        return 3;
    if (is("^"))
        return 4;
    if (is("&"))
        return 5;
    if (is("==") || is("!="))
        return 6;
    if (is("<") || is(">") || is("<=") || is(">="))
        return 7;
    if (is("<<") || is(">>"))
        return 8;
    if (is("+") || is("-"))
        return 9;
    if (is("*") || is("/") || is("%"))
        return 10;
    return -1;
}

bool Same(const char* a, const char* b) noexcept
{
    return std::strcmp(a, b) == 0;
}

// Image shape of a texture resource.
struct ImageShape
{
    u32 dim = 1;  // SPIR-V Dim: 0 1D, 1 2D, 2 3D, 3 Cube
    bool arrayed = false;
    bool ms = false;
    bool storage = false;
    u32 sizeComponents = 2;   // components GetDimensions reports, before levels
    u32 coordComponents = 2;  // components a sample/load coordinate has (without mip)
    bool valid = false;
};

ImageShape ShapeOf(Res kind) noexcept
{
    ImageShape s{};
    s.valid = true;
    switch (kind)
    {
    case Res::Texture1D:
    case Res::RWTexture1D:
        s.dim = 0;
        s.sizeComponents = 1;
        s.coordComponents = 1;
        break;
    case Res::Texture1DArray:
    case Res::RWTexture1DArray:
        s.dim = 0;
        s.arrayed = true;
        s.sizeComponents = 2;
        s.coordComponents = 2;
        break;
    case Res::Texture2D:
    case Res::RWTexture2D:
        break;
    case Res::Texture2DArray:
    case Res::RWTexture2DArray:
        s.arrayed = true;
        s.sizeComponents = 3;
        s.coordComponents = 3;
        break;
    case Res::Texture2DMS:
        s.ms = true;
        break;
    case Res::Texture3D:
    case Res::RWTexture3D:
        s.dim = 2;
        s.sizeComponents = 3;
        s.coordComponents = 3;
        break;
    case Res::TextureCube:
        s.dim = 3;
        s.sizeComponents = 2;
        s.coordComponents = 3;
        break;
    case Res::TextureCubeArray:
        s.dim = 3;
        s.arrayed = true;
        s.sizeComponents = 3;
        s.coordComponents = 4;
        break;
    default:
        s.valid = false;
        break;
    }
    s.storage = kind == Res::RWTexture1D || kind == Res::RWTexture1DArray ||
                kind == Res::RWTexture2D || kind == Res::RWTexture2DArray ||
                kind == Res::RWTexture3D;
    return s;
}

bool IsBufferResource(Res kind) noexcept
{
    return kind == Res::StructuredBuffer || kind == Res::RWStructuredBuffer ||
           kind == Res::ByteAddressBuffer || kind == Res::RWByteAddressBuffer;
}

}  // namespace

// ---- Helpers ------------------------------------------------------------------

TypeId Compiler::CommonType(const Val* values, u32 count, bool /*vectorMin*/) noexcept
{
    int rank = -1;
    u32 width = 0;  // 0 = all scalars
    for (u32 index = 0; index < count; ++index)
    {
        const TypeId type = values[index].type;
        if (!IsNumericScalarOrVector(type))
        {
            return 0;
        }
        const int r = Rank(T(ScalarOf(type)).kind);
        rank = r > rank ? r : rank;
        const u32 n = ComponentCount(type);
        if (n > 1)
        {
            width = width == 0 || n < width ? n : width;
        }
    }
    TypeId scalar = 0;
    switch (rank)
    {
    case 0:
        scalar = BoolType();
        break;
    case 1:
        scalar = IntType();
        break;
    case 2:
        scalar = UintType();
        break;
    case 3:
        scalar = FloatType();
        break;
    default:
        return 0;
    }
    return VectorOf(scalar, width == 0 ? 1 : width);
}

namespace
{

// Folds a binary operation on two constant scalars of `kind`.
bool FoldBinary(const char* op, TK kind, u32 a, u32 b, u32* out, bool* isBool) noexcept
{
    *isBool = false;
    const float fa = BitsFloat(a);
    const float fb = BitsFloat(b);
    const int ia = static_cast<int>(a);
    const int ib = static_cast<int>(b);
    const bool isFloat = kind == TK::Float;
    const bool isSigned = kind == TK::Int;
    const auto boolean = [&](bool value) noexcept
    {
        *isBool = true;
        *out = value ? 1U : 0U;
        return true;
    };
    if (Same(op, "<"))
        return boolean(isFloat ? fa < fb : isSigned ? ia < ib : a < b);
    if (Same(op, ">"))
        return boolean(isFloat ? fa > fb : isSigned ? ia > ib : a > b);
    if (Same(op, "<="))
        return boolean(isFloat ? fa <= fb : isSigned ? ia <= ib : a <= b);
    if (Same(op, ">="))
        return boolean(isFloat ? fa >= fb : isSigned ? ia >= ib : a >= b);
    if (Same(op, "=="))
        return boolean(isFloat ? fa == fb : a == b);
    if (Same(op, "!="))
        return boolean(isFloat ? !(fa == fb) : a != b);
    if (isFloat)
    {
        if (Same(op, "+"))
            *out = FloatBits(fa + fb);
        else if (Same(op, "-"))
            *out = FloatBits(fa - fb);
        else if (Same(op, "*"))
            *out = FloatBits(fa * fb);
        else if (Same(op, "/") && fb != 0.0F)
            *out = FloatBits(fa / fb);
        else
            return false;
        return true;
    }
    if (Same(op, "+"))
        *out = a + b;
    else if (Same(op, "-"))
        *out = a - b;
    else if (Same(op, "*"))
        *out = a * b;
    else if (Same(op, "/") && b != 0 && !(isSigned && ia == -2147483647 - 1 && ib == -1))
        *out = isSigned ? static_cast<u32>(ia / ib) : a / b;
    else if (Same(op, "%") && b != 0 && !(isSigned && ia == -2147483647 - 1 && ib == -1))
        *out = isSigned ? static_cast<u32>(ia % ib) : a % b;
    else if (Same(op, "&"))
        *out = a & b;
    else if (Same(op, "|"))
        *out = a | b;
    else if (Same(op, "^"))
        *out = a ^ b;
    else if (Same(op, "<<"))
        *out = a << (b & 31U);
    else if (Same(op, ">>"))
        *out = isSigned ? static_cast<u32>(ia >> (b & 31U)) : a >> (b & 31U);
    else
        return false;
    return true;
}

}  // namespace

// ---- Expressions ----------------------------------------------------------------

Val Compiler::ParseExpression() noexcept
{
    Val value = ParseAssignment();
    while (!m_error && IsPunct(","))
    {
        ++m_cursor;
        value = ParseAssignment();
    }
    return value;
}

Val Compiler::ParseAssignment() noexcept
{
    if (m_error)
    {
        return Val{};
    }
    const Val lhs = ParseTernary();
    const Token& opToken = Peek();
    static constexpr const char* kAssignOps[] = {
        "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="};
    const char* op = nullptr;
    for (const char* candidate : kAssignOps)
    {
        if (IsPunct(candidate))
        {
            op = candidate;
        }
    }
    if (op == nullptr || m_error)
    {
        return lhs;
    }
    ++m_cursor;
    const Val rhs = ParseAssignment();
    if (m_error)
    {
        return Val{};
    }
    TypeId target = 0;
    switch (lhs.kind)
    {
    case VK::Ptr:
        target = LogicalOf(lhs.type);
        break;
    case VK::Swz:
        target = VectorOf(ScalarOf(lhs.type), lhs.swizzleCount);
        break;
    case VK::Texel:
        target = lhs.type;
        break;
    default:
        (void)Fail(opToken, "the left side of an assignment is not assignable");
        return Val{};
    }
    if (lhs.readOnly)
    {
        (void)Fail(opToken, "cannot assign to a read-only value");
        return Val{};
    }
    Val value{};
    if (Same(op, "="))
    {
        value = Convert(rhs, target);
    }
    else
    {
        char binaryOp[3] = {};
        const usize length = std::strlen(op) - 1;
        std::memcpy(binaryOp, op, length);
        value = Convert(Binary(binaryOp, Load(lhs), rhs, opToken), target);
    }
    if (m_error || !Store(lhs, value))
    {
        return Val{};
    }
    return value;
}

Val Compiler::ParseTernary() noexcept
{
    const Val condition = ParseBinary(1);
    if (m_error || !IsPunct("?"))
    {
        return condition;
    }
    const Token& at = Next();
    Val condValue = ToBool(condition);
    if (m_error)
    {
        return Val{};
    }
    Val a{};
    Val b{};
    if (!m_atGlobalScope && IsScalar(condValue.type) && !condValue.isConst)
    {
        // A scalar condition evaluates only the chosen branch, as DXC does: the
        // other branch may read memory its condition guards. Each branch is
        // spilled in its own block and the results selected after the merge.
        const u32 thenLabel = NewId();
        const u32 elseLabel = NewId();
        const u32 merge = NewId();
        OpVoid(spv::OpSelectionMerge, {merge, 0});
        Terminate(spv::OpBranchConditional, {condValue.id, thenLabel, elseLabel});
        Label(thenLabel);
        const Val thenValue = Rvalue(ParseAssignment());
        if (m_error || !Expect(":", "expected ':' in a conditional expression"))
        {
            return Val{};
        }
        u32 thenSpill = 0;
        if (!thenValue.isConst)
        {
            thenSpill = NewLocal(thenValue.type);
            OpVoid(spv::OpStore, {thenSpill, thenValue.id});
        }
        Branch(merge);
        Label(elseLabel);
        const Val elseValue = Rvalue(ParseAssignment());
        if (m_error)
        {
            return Val{};
        }
        u32 elseSpill = 0;
        if (!elseValue.isConst)
        {
            elseSpill = NewLocal(elseValue.type);
            OpVoid(spv::OpStore, {elseSpill, elseValue.id});
        }
        Branch(merge);
        Label(merge);
        a = thenValue;
        b = elseValue;
        if (thenSpill != 0)
        {
            a.id = Op(spv::OpLoad, thenValue.type, {thenSpill});
        }
        if (elseSpill != 0)
        {
            b.id = Op(spv::OpLoad, elseValue.type, {elseSpill});
        }
    }
    else
    {
        a = Rvalue(ParseAssignment());
        if (!Expect(":", "expected ':' in a conditional expression"))
        {
            return Val{};
        }
        b = Rvalue(ParseAssignment());
        if (m_error)
        {
            return Val{};
        }
    }
    TypeId type = 0;
    if (a.type == b.type)
    {
        type = a.type;
    }
    else
    {
        const Val pair[2] = {a, b};
        type = CommonType(pair, 2, true);
        if (type == 0)
        {
            (void)Fail(at, "the branches of a conditional expression have incompatible types");
            return Val{};
        }
    }
    const Val left = Convert(a, type);
    const Val right = Convert(b, type);
    if (m_error)
    {
        return Val{};
    }
    if (condValue.isConst && left.isConst && right.isConst && IsScalar(type))
    {
        return condValue.constBits != 0 ? left : right;
    }
    const u32 condCount = ComponentCount(condValue.type);
    if (condCount > 1 && condCount != ComponentCount(type))
    {
        if (ComponentCount(type) == 1 || condCount < ComponentCount(type))
        {
            (void)Fail(at, "a vector condition must match the width of its branches");
            return Val{};
        }
        condValue = Convert(condValue, VectorOf(BoolType(), ComponentCount(type)));
    }
    Val result{};
    result.kind = VK::R;
    result.type = type;
    result.id = Op(spv::OpSelect, type, {condValue.id, left.id, right.id});
    return result;
}

Val Compiler::ParseBinary(int precedence) noexcept
{
    Val lhs = ParseUnary();
    for (;;)
    {
        if (m_error)
        {
            return Val{};
        }
        const Token& opToken = Peek();
        const int current = Precedence(opToken);
        if (current < precedence)
        {
            return lhs;
        }
        char op[3] = {};
        std::memcpy(op, opToken.text, opToken.length);
        ++m_cursor;
        const bool logical = Same(op, "&&") || Same(op, "||");
        if (logical && !m_atGlobalScope)
        {
            const Val left = ToBool(lhs);
            if (m_error)
            {
                return Val{};
            }
            if (IsScalar(left.type) && !left.isConst)
            {
                // Scalar && and || skip their right operand, as DXC does, so a
                // guard such as `i < n && data[i] > 0` never reads out of range.
                const bool isAnd = Same(op, "&&");
                const u32 rhsLabel = NewId();
                const u32 merge = NewId();
                const u32 spill = NewLocal(BoolType());
                OpVoid(spv::OpSelectionMerge, {merge, 0});
                Terminate(spv::OpBranchConditional,
                          {left.id, isAnd ? rhsLabel : merge, isAnd ? merge : rhsLabel});
                Label(rhsLabel);
                const Val right = ToBool(ParseBinary(current + 1));
                if (m_error)
                {
                    return Val{};
                }
                if (!IsScalar(right.type))
                {
                    (void)Fail(opToken, "mixing scalar and vector operands of && or ||");
                    return Val{};
                }
                OpVoid(spv::OpStore, {spill, right.id});
                Branch(merge);
                Label(merge);
                const u32 loaded = Op(spv::OpLoad, BoolType(), {spill});
                Val result{};
                result.kind = VK::R;
                result.type = BoolType();
                result.id = isAnd ? Op(spv::OpSelect, BoolType(),
                                       {left.id, loaded, ConstantBits(BoolType(), 0)})
                                  : Op(spv::OpSelect, BoolType(),
                                       {left.id, ConstantBits(BoolType(), 1), loaded});
                lhs = result;
                continue;
            }
            lhs = left;
        }
        const Val rhs = ParseBinary(current + 1);
        lhs = Binary(op, lhs, rhs, opToken);
    }
}

Val Compiler::Binary(const char* op, Val lhsIn, Val rhsIn, const Token& at) noexcept
{
    if (m_error)
    {
        return Val{};
    }
    const Val lhs = Rvalue(lhsIn);
    const Val rhs = Rvalue(rhsIn);
    if (m_error)
    {
        return Val{};
    }
    if (Same(op, "+") || Same(op, "-") || Same(op, "*") || Same(op, "/") || Same(op, "%"))
    {
        return Arithmetic(op, lhs, rhs, at);
    }
    const Val pair[2] = {lhs, rhs};
    const bool logical = Same(op, "&&") || Same(op, "||");
    const bool compare = Same(op, "<") || Same(op, ">") || Same(op, "<=") || Same(op, ">=") ||
                         Same(op, "==") || Same(op, "!=");
    const bool shift = Same(op, "<<") || Same(op, ">>");
    TypeId type = CommonType(pair, 2, true);
    if (type == 0)
    {
        (void)Fail(at, "invalid operand types for a binary operator");
        return Val{};
    }
    if (logical)
    {
        type = WithScalar(type, BoolType());
    }
    else if (shift)
    {
        if (!IsInteger(lhs.type) && !IsBoolish(lhs.type))
        {
            (void)Fail(at, "shifts need integer operands");
            return Val{};
        }
        type = WithScalar(type, IsBoolish(lhs.type) ? IntType() : ScalarOf(lhs.type));
    }
    else if (!compare)
    {
        // & | ^
        if (IsFloatish(type))
        {
            (void)Fail(at, "bitwise operators need integer or bool operands");
            return Val{};
        }
    }
    const Val a = Convert(lhs, type);
    const Val b = Convert(rhs, type);
    if (m_error)
    {
        return Val{};
    }
    const TK kind = T(ScalarOf(type)).kind;
    const TypeId boolType = WithScalar(type, BoolType());
    if (a.isConst && b.isConst && IsScalar(type))
    {
        u32 bits = 0;
        bool isBool = false;
        if (logical)
        {
            const bool value =
                Same(op, "&&") ? (a.constBits && b.constBits) : (a.constBits || b.constBits);
            return ConstVal(BoolType(), value ? 1U : 0U);
        }
        if (kind == TK::Bool && !compare)
        {
            const u32 value = Same(op, "&")   ? (a.constBits & b.constBits)
                              : Same(op, "|") ? (a.constBits | b.constBits)
                                              : (a.constBits ^ b.constBits);
            return ConstVal(BoolType(), value);
        }
        if (FoldBinary(op, kind == TK::Bool ? TK::Uint : kind, a.constBits, b.constBits, &bits,
                       &isBool))
        {
            return ConstVal(isBool ? BoolType() : type, bits);
        }
    }
    Val result{};
    result.kind = VK::R;
    if (logical)
    {
        result.type = type;
        result.id = Op(Same(op, "&&") ? spv::OpLogicalAnd : spv::OpLogicalOr, type, {a.id, b.id});
        return result;
    }
    if (compare)
    {
        u32 opcode = 0;
        if (kind == TK::Bool)
        {
            if (Same(op, "=="))
                opcode = spv::OpLogicalEqual;
            else if (Same(op, "!="))
                opcode = spv::OpLogicalNotEqual;
            else
            {
                (void)Fail(at, "bools can only be compared for equality");
                return Val{};
            }
        }
        else if (kind == TK::Float)
        {
            opcode = Same(op, "<")    ? spv::OpFOrdLessThan
                     : Same(op, ">")  ? spv::OpFOrdGreaterThan
                     : Same(op, "<=") ? spv::OpFOrdLessThanEqual
                     : Same(op, ">=") ? spv::OpFOrdGreaterThanEqual
                     : Same(op, "==") ? spv::OpFOrdEqual
                                      : spv::OpFUnordNotEqual;
        }
        else
        {
            const bool s = kind == TK::Int;
            opcode = Same(op, "<")    ? (s ? spv::OpSLessThan : spv::OpULessThan)
                     : Same(op, ">")  ? (s ? spv::OpSGreaterThan : spv::OpUGreaterThan)
                     : Same(op, "<=") ? (s ? spv::OpSLessThanEqual : spv::OpULessThanEqual)
                     : Same(op, ">=") ? (s ? spv::OpSGreaterThanEqual : spv::OpUGreaterThanEqual)
                     : Same(op, "==") ? spv::OpIEqual
                                      : spv::OpINotEqual;
        }
        result.type = boolType;
        result.id = Op(opcode, boolType, {a.id, b.id});
        return result;
    }
    result.type = type;
    if (kind == TK::Bool)
    {
        const u32 opcode = Same(op, "&")   ? spv::OpLogicalAnd
                           : Same(op, "|") ? spv::OpLogicalOr
                                           : spv::OpLogicalNotEqual;
        result.id = Op(opcode, type, {a.id, b.id});
        return result;
    }
    if (shift)
    {
        // D3D uses the low five bits of the shift amount.
        u32 amount = b.id;
        if (!b.isConst)
        {
            u32 mask = ConstantBits(ScalarOf(type), 31);
            const u32 count = ComponentCount(type);
            if (count > 1)
            {
                u32 masks[4] = {mask, mask, mask, mask};
                mask = ConstantComposite(type, masks, count);
            }
            amount = Op(spv::OpBitwiseAnd, type, {b.id, mask});
        }
        else
        {
            const Val masked = ConstVal(ScalarOf(type), b.constBits & 31U);
            amount = IsScalar(type) ? masked.id : Splat(masked, type).id;
        }
        const u32 opcode = Same(op, "<<")    ? spv::OpShiftLeftLogical
                           : kind == TK::Int ? spv::OpShiftRightArithmetic
                                             : spv::OpShiftRightLogical;
        result.id = Op(opcode, type, {a.id, amount});
        return result;
    }
    const u32 opcode = Same(op, "&")   ? spv::OpBitwiseAnd
                       : Same(op, "|") ? spv::OpBitwiseOr
                                       : spv::OpBitwiseXor;
    result.id = Op(opcode, type, {a.id, b.id});
    return result;
}

Val Compiler::Arithmetic(const char* op, Val lhs, Val rhs, const Token& at) noexcept
{
    lhs = Rvalue(lhs);
    rhs = Rvalue(rhs);
    if (m_error)
    {
        return Val{};
    }
    const bool lm = T(lhs.type).kind == TK::Matrix;
    const bool rm = T(rhs.type).kind == TK::Matrix;
    Val result{};
    result.kind = VK::R;
    if (lm || rm)
    {
        if (Same(op, "*") && lm != rm)
        {
            const Val matrix = lm ? lhs : rhs;
            const Val scalar = Convert(lm ? rhs : lhs, FloatType());
            if (m_error)
            {
                return Val{};
            }
            if (!IsScalar(scalar.type))
            {
                (void)Fail(at, "use mul() for matrix-vector products");
                return Val{};
            }
            result.type = matrix.type;
            result.id = Op(spv::OpMatrixTimesScalar, matrix.type, {matrix.id, scalar.id});
            return result;
        }
        if (lm && rm && lhs.type == rhs.type && !Same(op, "%"))
        {
            const TypeEntry m = T(lhs.type);
            u32 columns[4] = {};
            const u32 opcode = Same(op, "+")   ? spv::OpFAdd
                               : Same(op, "-") ? spv::OpFSub
                               : Same(op, "*") ? spv::OpFMul
                                               : spv::OpFDiv;
            for (u32 column = 0; column < m.b; ++column)
            {
                const u32 x = Op(spv::OpCompositeExtract, m.a, {lhs.id, column});
                const u32 y = Op(spv::OpCompositeExtract, m.a, {rhs.id, column});
                columns[column] = Op(opcode, m.a, {x, y});
            }
            result.type = lhs.type;
            result.id = OpN(spv::OpCompositeConstruct, lhs.type, columns, m.b);
            return result;
        }
        (void)Fail(at, "this matrix arithmetic is not supported by the Vulkan HLSL compiler");
        return Val{};
    }
    const Val pair[2] = {lhs, rhs};
    TypeId type = CommonType(pair, 2, true);
    if (type == 0)
    {
        (void)Fail(at, "invalid operand types for an arithmetic operator");
        return Val{};
    }
    if (IsBoolish(type))
    {
        type = WithScalar(type, IntType());
    }
    const Val a = Convert(lhs, type);
    const Val b = Convert(rhs, type);
    if (m_error)
    {
        return Val{};
    }
    const TK kind = T(ScalarOf(type)).kind;
    if (a.isConst && b.isConst && IsScalar(type))
    {
        u32 bits = 0;
        bool isBool = false;
        if (FoldBinary(op, kind, a.constBits, b.constBits, &bits, &isBool))
        {
            return ConstVal(type, bits);
        }
    }
    u32 opcode = 0;
    if (kind == TK::Float)
    {
        opcode = Same(op, "+")   ? spv::OpFAdd
                 : Same(op, "-") ? spv::OpFSub
                 : Same(op, "*") ? spv::OpFMul
                 : Same(op, "/") ? spv::OpFDiv
                                 : spv::OpFRem;
    }
    else
    {
        const bool s = kind == TK::Int;
        opcode = Same(op, "+")   ? spv::OpIAdd
                 : Same(op, "-") ? spv::OpISub
                 : Same(op, "*") ? spv::OpIMul
                 : Same(op, "/") ? (s ? spv::OpSDiv : spv::OpUDiv)
                                 : (s ? spv::OpSRem : spv::OpUMod);
    }
    result.type = type;
    result.id = Op(opcode, type, {a.id, b.id});
    if (m_precise && kind == TK::Float)
    {
        Decorate(result.id, spv::DecNoContraction);
    }
    return result;
}

Val Compiler::ParseUnary() noexcept
{
    if (m_depth >= 64)
    {
        (void)FailHere("expression is nested too deeply");
        return Val{};
    }
    ++m_depth;
    const Val value = ParseUnaryInner();
    --m_depth;
    return value;
}

Val Compiler::ParseUnaryInner() noexcept
{
    if (m_error)
    {
        return Val{};
    }
    const Token& at = Peek();
    if (IsPunct("-") || IsPunct("+") || IsPunct("!") || IsPunct("~"))
    {
        char op = at.text[0];
        ++m_cursor;
        Val value = Rvalue(ParseUnary());
        if (m_error)
        {
            return Val{};
        }
        if (op == '+')
        {
            return value;
        }
        if (op == '!')
        {
            value = ToBool(value);
            if (value.isConst && IsScalar(value.type))
            {
                return ConstVal(BoolType(), value.constBits != 0 ? 0U : 1U);
            }
            Val result{};
            result.kind = VK::R;
            result.type = value.type;
            result.id = Op(spv::OpLogicalNot, value.type, {value.id});
            return result;
        }
        if (!IsNumericScalarOrVector(value.type))
        {
            if (op == '-' && T(value.type).kind == TK::Matrix)
            {
                Val result{};
                result.kind = VK::R;
                result.type = value.type;
                result.id = Op(spv::OpMatrixTimesScalar, value.type, {value.id, ConstFloat(-1.0F)});
                return result;
            }
            (void)Fail(at, "invalid operand for a unary operator");
            return Val{};
        }
        if (IsBoolish(value.type))
        {
            value = Convert(value, WithScalar(value.type, IntType()));
        }
        const TK kind = T(ScalarOf(value.type)).kind;
        if (op == '~' && kind == TK::Float)
        {
            (void)Fail(at, "~ needs an integer operand");
            return Val{};
        }
        if (value.isConst && IsScalar(value.type))
        {
            if (op == '~')
            {
                return ConstVal(value.type, ~value.constBits);
            }
            if (kind == TK::Float)
            {
                return ConstVal(value.type, value.constBits ^ 0x80000000U);
            }
            return ConstVal(value.type, 0U - value.constBits);
        }
        Val result{};
        result.kind = VK::R;
        result.type = value.type;
        const u32 opcode = op == '~'           ? spv::OpNot
                           : kind == TK::Float ? spv::OpFNegate
                                               : spv::OpSNegate;
        result.id = Op(opcode, value.type, {value.id});
        return result;
    }
    if (IsPunct("++") || IsPunct("--"))
    {
        const bool increment = IsPunct("++");
        ++m_cursor;
        const Val target = ParseUnary();
        const Val current = Load(target);
        if (m_error)
        {
            return Val{};
        }
        const Val one = ConstVal(IntType(), 1);
        const Val next = Convert(Binary(increment ? "+" : "-", current, one, at), current.type);
        if (!Store(target, next))
        {
            return Val{};
        }
        return next;
    }
    // A cast: `(type)` followed by a unary expression.
    if (IsPunct("(") && IsTypeStart(1))
    {
        u32 close = 2;
        // Allow array casts like (float[4]) to fail clearly below.
        if (IsPunct(")", close))
        {
            ++m_cursor;
            const TypeId type = ParseType(false);
            if (type == 0 || !Expect(")", "expected ')'"))
            {
                return Val{};
            }
            const Val value = Rvalue(ParseUnary());
            if (m_error)
            {
                return Val{};
            }
            if (T(type).kind == TK::Struct || T(type).kind == TK::Array)
            {
                if (value.isConst && IsScalar(value.type) && value.constBits == 0)
                {
                    Val zero{};
                    zero.kind = VK::R;
                    zero.type = type;
                    zero.id = ConstantNull(type);
                    zero.isConst = true;
                    return zero;
                }
                (void)Fail(at, "only (Struct)0 is supported as a structure cast");
                return Val{};
            }
            if (IsNumericScalarOrVector(type) && IsNumericScalarOrVector(value.type) &&
                ComponentCount(value.type) == 1 && ComponentCount(type) > 1)
            {
                return Convert(value, type);  // splat
            }
            if (IsNumericScalarOrVector(type) && IsNumericScalarOrVector(value.type) &&
                ComponentCount(value.type) < ComponentCount(type))
            {
                (void)Fail(at, "cannot cast a vector to a wider vector");
                return Val{};
            }
            return Convert(value, type);
        }
    }
    return ParsePostfix(ParsePrimary());
}

Val Compiler::ParsePostfix(Val value) noexcept
{
    for (;;)
    {
        if (m_error)
        {
            return Val{};
        }
        const Token& at = Peek();
        if (IsPunct("."))
        {
            ++m_cursor;
            const Name name = ExpectIdent("expected a member name");
            if (m_error)
            {
                return Val{};
            }
            if (IsPunct("("))
            {
                Val args[8] = {};
                u32 count = 0;
                if (!ParseArguments(args, 8, &count))
                {
                    return Val{};
                }
                value = CallMethod(value, name, args, count, at);
            }
            else
            {
                value = Member(value, name, at);
            }
            continue;
        }
        if (IsPunct("["))
        {
            ++m_cursor;
            const Val index = ParseExpression();
            if (!Expect("]", "expected ']'"))
            {
                return Val{};
            }
            value = Index(value, index, at);
            continue;
        }
        if (IsPunct("++") || IsPunct("--"))
        {
            const bool increment = IsPunct("++");
            ++m_cursor;
            const Val current = Load(value);
            if (m_error)
            {
                return Val{};
            }
            const Val next = Convert(
                Binary(increment ? "+" : "-", current, ConstVal(IntType(), 1), at), current.type);
            if (!Store(value, next))
            {
                return Val{};
            }
            return current;
        }
        return value;
    }
}

bool Compiler::ParseArguments(Val* args, u32 capacity, u32* count) noexcept
{
    *count = 0;
    if (!Expect("(", "expected '('"))
    {
        return false;
    }
    if (Accept(")"))
    {
        return true;
    }
    for (;;)
    {
        if (*count == capacity)
        {
            return FailHere("too many arguments");
        }
        args[(*count)++] = ParseAssignment();
        if (m_error)
        {
            return false;
        }
        if (Accept(")"))
        {
            return true;
        }
        if (!Expect(",", "expected ',' or ')' in an argument list"))
        {
            return false;
        }
    }
}

Val Compiler::ParsePrimary() noexcept
{
    if (m_error)
    {
        return Val{};
    }
    const Token& token = Peek();
    if (token.kind == Tok::Int)
    {
        ++m_cursor;
        return ConstVal(token.isUnsigned ? UintType() : IntType(),
                        static_cast<u32>(token.intValue));
    }
    if (token.kind == Tok::Float)
    {
        ++m_cursor;
        return ConstVal(FloatType(), FloatBits(static_cast<float>(token.floatValue)));
    }
    if (IsPunct("("))
    {
        ++m_cursor;
        const Val value = ParseExpression();
        if (!Expect(")", "expected ')'"))
        {
            return Val{};
        }
        return value;
    }
    if (token.kind != Tok::Ident)
    {
        (void)FailHere("expected an expression");
        return Val{};
    }
    const Name name{token.text, token.length};
    if (name.Is("true") || name.Is("false"))
    {
        ++m_cursor;
        return ConstVal(BoolType(), name.Is("true") ? 1U : 0U);
    }
    // Constructors.
    if (IsTypeStart(0))
    {
        const u32 start = m_cursor;
        const TypeId type = ParseType(false);
        if (m_error)
        {
            return Val{};
        }
        if (IsPunct("("))
        {
            Val args[16] = {};
            u32 count = 0;
            if (!ParseArguments(args, 16, &count))
            {
                return Val{};
            }
            return Construct(type, args, count, token);
        }
        m_cursor = start;
    }
    const Symbol* symbol = Lookup(name);
    if (symbol != nullptr && !IsPunct("(", 1))
    {
        ++m_cursor;
        switch (symbol->kind)
        {
        case SK::Var:
        case SK::Const:
            return symbol->value;
        case SK::CBufferMember:
            return CBufferMember(symbol->index, symbol->member, token);
        case SK::Resource:
        {
            Val value{};
            value.kind = VK::Resource;
            value.resource = symbol->index + 1;
            return value;
        }
        default:
            (void)Fail(token, "a type name is not a value");
            return Val{};
        }
    }
    if (IsPunct("(", 1))
    {
        ++m_cursor;
        Val args[16] = {};
        u32 count = 0;
        if (!ParseArguments(args, 16, &count))
        {
            return Val{};
        }
        for (u32 index = 0; index < m_funcs.Size(); ++index)
        {
            if (m_funcs[index].name.Equals(name))
            {
                return CallFunction(name, args, count, token);
            }
        }
        bool handled = false;
        const Val result = CallIntrinsic(name, args, count, token, &handled);
        if (!handled && !m_error)
        {
            (void)Fail(token, "unknown function or intrinsic (not supported by the Vulkan HLSL "
                              "compiler)");
            return Val{};
        }
        return result;
    }
    (void)Fail(token, "undeclared identifier");
    return Val{};
}

// ---- Composites -------------------------------------------------------------------

bool Compiler::FlattenScalars(const Val& valueIn, Vec<Val>& out) noexcept
{
    const Val value = Rvalue(valueIn);
    if (m_error)
    {
        return false;
    }
    const TypeEntry entry = T(value.type);
    if (IsScalar(value.type))
    {
        return out.Push(value);
    }
    u32 count = 0;
    TypeId elementType = 0;
    switch (entry.kind)
    {
    case TK::Vector:
        count = entry.b;
        elementType = entry.a;
        break;
    case TK::Matrix:
        count = entry.b;
        elementType = entry.a;
        break;
    case TK::Array:
        count = entry.b;
        elementType = entry.a;
        break;
    case TK::Struct:
    {
        const StructDef& def = m_structs[entry.a];
        for (u32 member = 0; member < def.memberCount; ++member)
        {
            const TypeId memberType = LaidOut(m_members[def.firstMember + member].type,
                                              static_cast<Layout>(entry.b), false);
            Val part{};
            part.kind = VK::R;
            part.type = memberType;
            part.id = Op(spv::OpCompositeExtract, memberType, {value.id, member});
            if (!FlattenScalars(part, out))
            {
                return false;
            }
        }
        return true;
    }
    default:
        return FailHere("this value cannot be used in an initializer");
    }
    // Constant composites flatten without instructions, so global initialisers work.
    const ConstKey* constant = nullptr;
    if (value.isConst)
    {
        for (u32 index = 0; index < m_constants.Size(); ++index)
        {
            if (m_constants[index].id == value.id)
            {
                constant = &m_constants[index];
                break;
            }
        }
    }
    for (u32 index = 0; index < count; ++index)
    {
        Val part{};
        part.kind = VK::R;
        part.type = elementType;
        if (constant != nullptr && constant->op == spv::OpConstantComposite)
        {
            part.id = m_constOperands[constant->first + index];
            part.isConst = true;
            for (u32 search = 0; search < m_constants.Size(); ++search)
            {
                const ConstKey& key = m_constants[search];
                if (key.id == part.id && IsScalar(elementType))
                {
                    part.constBits = key.op == spv::OpConstant       ? m_constOperands[key.first]
                                     : key.op == spv::OpConstantTrue ? 1U
                                                                     : 0U;
                }
            }
        }
        else
        {
            part.id = Op(spv::OpCompositeExtract, elementType, {value.id, index});
        }
        if (!FlattenScalars(part, out))
        {
            return false;
        }
    }
    return true;
}

Val Compiler::BuildFromScalars(TypeId type, const Vec<Val>& scalars, u32* cursor,
                               const Token& at) noexcept
{
    if (m_error)
    {
        return Val{};
    }
    if (IsScalar(type))
    {
        if (*cursor >= scalars.Size())
        {
            (void)Fail(at, "too few values in an initializer or constructor");
            return Val{};
        }
        return Convert(scalars[(*cursor)++], type);
    }
    const TypeEntry entry = T(type);
    u32 count = 0;
    TypeId elementTypes[64] = {};
    switch (entry.kind)
    {
    case TK::Vector:
    case TK::Matrix:
    case TK::Array:
        count = entry.b;
        if (count > 64 && entry.kind == TK::Array)
        {
            (void)Fail(at, "initializers are limited to 64 array elements");
            return Val{};
        }
        for (u32 index = 0; index < count; ++index)
        {
            elementTypes[index] = entry.a;
        }
        break;
    case TK::Struct:
    {
        const StructDef& def = m_structs[entry.a];
        count = def.memberCount;
        for (u32 member = 0; member < count; ++member)
        {
            elementTypes[member] = LaidOut(m_members[def.firstMember + member].type,
                                           static_cast<Layout>(entry.b), false);
        }
        break;
    }
    default:
        (void)Fail(at, "this type cannot be initialised from a list");
        return Val{};
    }
    u32 ids[64] = {};
    bool allConst = true;
    for (u32 index = 0; index < count; ++index)
    {
        const Val part = BuildFromScalars(elementTypes[index], scalars, cursor, at);
        if (m_error)
        {
            return Val{};
        }
        ids[index] = part.id;
        allConst = allConst && part.isConst;
    }
    Val result{};
    result.kind = VK::R;
    result.type = type;
    if (allConst)
    {
        result.id = ConstantComposite(type, ids, count);
        result.isConst = true;
    }
    else
    {
        result.id = OpN(spv::OpCompositeConstruct, type, ids, count);
    }
    return result;
}

Val Compiler::ParseInitializer(TypeId type) noexcept
{
    const Token& at = Peek();
    Vec<Val> scalars;
    scalars.Init(&m_arena);
    if (!Expect("{", "expected '{'"))
    {
        return Val{};
    }
    u32 depth = 1;
    while (!m_error && depth > 0)
    {
        if (Accept("{"))
        {
            ++depth;
            continue;
        }
        if (Accept("}"))
        {
            --depth;
            (void)Accept(",");
            continue;
        }
        if (Peek().kind == Tok::End)
        {
            (void)FailHere("unterminated initializer");
            return Val{};
        }
        if (!FlattenScalars(ParseAssignment(), scalars))
        {
            return Val{};
        }
        if (!IsPunct("}") && !Expect(",", "expected ',' in an initializer"))
        {
            return Val{};
        }
    }
    u32 cursor = 0;
    const Val result = BuildFromScalars(type, scalars, &cursor, at);
    if (!m_error && cursor != scalars.Size())
    {
        (void)Fail(at, "too many values in an initializer");
        return Val{};
    }
    return result;
}

Val Compiler::Construct(TypeId type, const Val* args, u32 count, const Token& at) noexcept
{
    if (m_error)
    {
        return Val{};
    }
    if (count == 0)
    {
        (void)Fail(at, "a constructor needs arguments");
        return Val{};
    }
    if (IsScalar(type))
    {
        if (count != 1)
        {
            (void)Fail(at, "a scalar constructor takes one argument");
            return Val{};
        }
        return Convert(args[0], type);
    }
    const TypeEntry entry = T(type);
    if (entry.kind != TK::Vector && entry.kind != TK::Matrix)
    {
        (void)Fail(at, "only vector and matrix constructors exist; use an initializer list");
        return Val{};
    }
    if (count == 1)
    {
        const Val only = Rvalue(args[0]);
        if (m_error)
        {
            return Val{};
        }
        if (IsScalar(only.type))
        {
            if (entry.kind == TK::Matrix)
            {
                const Val scalar = Convert(only, FloatType());
                const Val column = Splat(scalar, entry.a);
                u32 columns[4] = {column.id, column.id, column.id, column.id};
                Val result{};
                result.kind = VK::R;
                result.type = type;
                result.isConst = scalar.isConst;
                result.id = scalar.isConst ? ConstantComposite(type, columns, entry.b)
                                           : OpN(spv::OpCompositeConstruct, type, columns, entry.b);
                return result;
            }
            return Convert(only, type);
        }
        if (only.type == type)
        {
            return only;
        }
    }
    if (entry.kind == TK::Vector)
    {
        // Constituents may be scalars or vectors of the element type.
        u32 total = 0;
        u32 ids[16] = {};
        bool allConst = true;
        bool allScalar = true;
        for (u32 index = 0; index < count; ++index)
        {
            const Val arg = Rvalue(args[index]);
            if (m_error)
            {
                return Val{};
            }
            if (!IsNumericScalarOrVector(arg.type))
            {
                (void)Fail(at, "invalid vector constructor argument");
                return Val{};
            }
            const u32 n = ComponentCount(arg.type);
            const Val converted = Convert(arg, WithScalar(arg.type, entry.a));
            if (m_error)
            {
                return Val{};
            }
            ids[index] = converted.id;
            allConst = allConst && converted.isConst;
            allScalar = allScalar && n == 1;
            total += n;
        }
        if (total != entry.b)
        {
            (void)Fail(at, "vector constructor component count mismatch");
            return Val{};
        }
        Val result{};
        result.kind = VK::R;
        result.type = type;
        if (allConst && allScalar)
        {
            result.id = ConstantComposite(type, ids, count);
            result.isConst = true;
            return result;
        }
        if (allConst)
        {
            // Constant vectors among the arguments: flatten so the result folds.
            Vec<Val> scalars;
            scalars.Init(&m_arena);
            for (u32 index = 0; index < count; ++index)
            {
                if (!FlattenScalars(
                        Convert(args[index], WithScalar(Rvalue(args[index]).type, entry.a)),
                        scalars))
                {
                    return Val{};
                }
            }
            u32 cursor = 0;
            return BuildFromScalars(type, scalars, &cursor, at);
        }
        result.id = OpN(spv::OpCompositeConstruct, type, ids, count);
        return result;
    }
    // Matrix: rows as vectors, or every element as scalars.
    Vec<Val> scalars;
    scalars.Init(&m_arena);
    for (u32 index = 0; index < count; ++index)
    {
        if (!FlattenScalars(args[index], scalars))
        {
            return Val{};
        }
    }
    if (scalars.Size() != ComponentCount(type))
    {
        (void)Fail(at, "matrix constructor component count mismatch");
        return Val{};
    }
    u32 cursor = 0;
    return BuildFromScalars(type, scalars, &cursor, at);
}

// ---- Member access, swizzles and indexing -------------------------------------------

Val Compiler::Swizzle(const Val& value, const Name& name, const Token& at) noexcept
{
    const TypeId type =
        value.kind == VK::Swz ? VectorOf(ScalarOf(value.type), value.swizzleCount) : value.type;
    const u32 width = ComponentCount(type);
    if (name.length == 0 || name.length > 4)
    {
        (void)Fail(at, "invalid swizzle");
        return Val{};
    }
    u8 components[4] = {};
    bool rgba = false;
    bool xyzw = false;
    bool duplicate = false;
    for (u32 index = 0; index < name.length; ++index)
    {
        const char c = name.text[index];
        int component = -1;
        switch (c)
        {
        case 'x':
            component = 0;
            xyzw = true;
            break;
        case 'y':
            component = 1;
            xyzw = true;
            break;
        case 'z':
            component = 2;
            xyzw = true;
            break;
        case 'w':
            component = 3;
            xyzw = true;
            break;
        case 'r':
            component = 0;
            rgba = true;
            break;
        case 'g':
            component = 1;
            rgba = true;
            break;
        case 'b':
            component = 2;
            rgba = true;
            break;
        case 'a':
            component = 3;
            rgba = true;
            break;
        default:
            break;
        }
        if (component < 0 || static_cast<u32>(component) >= width || (rgba && xyzw))
        {
            (void)Fail(at, "invalid swizzle or member name");
            return Val{};
        }
        for (u32 prior = 0; prior < index; ++prior)
        {
            duplicate = duplicate || components[prior] == component;
        }
        components[index] = static_cast<u8>(component);
    }
    const TypeId scalar = ScalarOf(type);
    if (value.kind == VK::Ptr && T(value.type).kind == TK::Vector)
    {
        if (name.length == 1)
        {
            Val result = value;
            result.type = scalar;
            result.id = Op(spv::OpAccessChain, PointerTo(value.storage, scalar),
                           {value.id, ConstUint(components[0])});
            return result;
        }
        if (!duplicate)
        {
            Val result = value;
            result.kind = VK::Swz;
            result.swizzleCount = static_cast<u8>(name.length);
            for (u32 index = 0; index < name.length; ++index)
            {
                result.swizzle[index] = components[index];
            }
            return result;
        }
    }
    if (value.kind == VK::Ptr && IsScalar(value.type) && !duplicate && name.length == 1)
    {
        return value;  // `scalar.x` on a variable is still the variable
    }
    const Val loaded = Rvalue(value);
    if (m_error)
    {
        return Val{};
    }
    Val result{};
    result.kind = VK::R;
    if (IsScalar(loaded.type))
    {
        result.type = VectorOf(scalar, name.length);
        result.isConst = loaded.isConst;
        result.constBits = loaded.constBits;
        if (name.length == 1)
        {
            return loaded;
        }
        return Splat(loaded, result.type);
    }
    if (name.length == 1)
    {
        result.type = scalar;
        result.id = Op(spv::OpCompositeExtract, scalar, {loaded.id, components[0]});
        return result;
    }
    result.type = VectorOf(scalar, name.length);
    u32 operands[6] = {loaded.id, loaded.id};
    for (u32 index = 0; index < name.length; ++index)
    {
        operands[2 + index] = components[index];
    }
    result.id = OpN(spv::OpVectorShuffle, result.type, operands, 2 + name.length);
    return result;
}

Val Compiler::Member(const Val& object, const Name& member, const Token& at) noexcept
{
    if (m_error)
    {
        return Val{};
    }
    if (object.kind == VK::Resource)
    {
        const ResourceDecl& decl = m_resources[object.resource - 1];
        if (decl.kind != Res::CBuffer || decl.cbufferMembersGlobal)
        {
            (void)Fail(at, "resources have methods, not members");
            return Val{};
        }
        const StructDef& def = m_structs[decl.structDef];
        for (u32 index = 0; index < def.memberCount; ++index)
        {
            if (!m_members[def.firstMember + index].name.Equals(member))
            {
                continue;
            }
            // ConstantBuffer<T> name, or an element of an indexed array of them.
            const Val block = ResourceElement(object, at);
            if (m_error)
            {
                return Val{};
            }
            const ResourceDecl& bound = m_resources[object.resource - 1];
            const TypeId memberType =
                LaidOut(m_members[def.firstMember + index].type, LayoutCBuffer, false);
            Val result{};
            result.kind = VK::Ptr;
            result.type = memberType;
            result.storage = bound.storage;
            result.readOnly = true;
            result.id = Op(spv::OpAccessChain, PointerTo(bound.storage, memberType),
                           {block.id, ConstUint(index)});
            return result;
        }
        (void)Fail(at, "no such member in the constant buffer");
        return Val{};
    }
    const TypeEntry entry = T(object.kind == VK::Swz ? 0 : object.type);
    if (object.kind != VK::Swz && entry.kind == TK::Struct)
    {
        const StructDef& def = m_structs[entry.a];
        for (u32 index = 0; index < def.memberCount; ++index)
        {
            const StructMember& info = m_members[def.firstMember + index];
            if (!info.name.Equals(member))
            {
                continue;
            }
            const TypeId memberType = LaidOut(info.type, static_cast<Layout>(entry.b), false);
            Val result = object;
            result.type = memberType;
            if (object.kind == VK::Ptr)
            {
                result.id = Op(spv::OpAccessChain, PointerTo(object.storage, memberType),
                               {object.id, ConstUint(index)});
            }
            else
            {
                const Val loaded = Rvalue(object);
                result = Val{};
                result.kind = VK::R;
                result.type = memberType;
                result.id = Op(spv::OpCompositeExtract, memberType, {loaded.id, index});
            }
            return result;
        }
        (void)Fail(at, "no such member");
        return Val{};
    }
    if (object.kind != VK::Swz && entry.kind == TK::Matrix && member.length >= 3 &&
        member.text[0] == '_')
    {
        u32 row = 0;
        u32 column = 0;
        if (member.length == 4 && member.text[1] == 'm')
        {
            row = static_cast<u32>(member.text[2] - '0');
            column = static_cast<u32>(member.text[3] - '0');
        }
        else if (member.length == 3)
        {
            row = static_cast<u32>(member.text[1] - '1');
            column = static_cast<u32>(member.text[2] - '1');
        }
        else
        {
            (void)Fail(at, "multi-element matrix swizzles are not supported");
            return Val{};
        }
        if (row >= entry.b || column >= T(entry.a).b)
        {
            (void)Fail(at, "matrix element out of range");
            return Val{};
        }
        const TypeId scalar = FloatType();
        Val result = object;
        result.type = scalar;
        if (object.kind == VK::Ptr)
        {
            result.id = Op(spv::OpAccessChain, PointerTo(object.storage, scalar),
                           {object.id, ConstUint(row), ConstUint(column)});
        }
        else
        {
            const Val loaded = Rvalue(object);
            result = Val{};
            result.kind = VK::R;
            result.type = scalar;
            result.id = Op(spv::OpCompositeExtract, scalar, {loaded.id, row, column});
        }
        return result;
    }
    return Swizzle(object, member, at);
}

Val Compiler::Index(const Val& object, Val indexIn, const Token& at) noexcept
{
    if (m_error)
    {
        return Val{};
    }
    Val index = Rvalue(indexIn);
    if (m_error)
    {
        return Val{};
    }
    if (object.kind == VK::Resource)
    {
        ResourceDecl& decl = m_resources[object.resource - 1];
        if (decl.arraySize != 0 && object.aux == 0)
        {
            index = Convert(index, UintType());
            if (m_error)
            {
                return Val{};
            }
            Val result = object;
            result.aux = index.id;
            result.nonUniform = indexIn.nonUniform || index.nonUniform;
            return result;
        }
        if (!IsBufferResource(decl.kind) && !ShapeOf(decl.kind).valid)
        {
            (void)Fail(at, "this resource cannot be indexed");
            return Val{};
        }
        if (decl.kind == Res::ByteAddressBuffer || decl.kind == Res::RWByteAddressBuffer)
        {
            (void)Fail(at, "byte address buffers are read with Load and written with Store");
            return Val{};
        }
        if (decl.kind == Res::StructuredBuffer || decl.kind == Res::RWStructuredBuffer)
        {
            const Val block = ResourceElement(object, at);
            if (m_error)
            {
                return Val{};
            }
            index = Convert(index, UintType());
            const TypeId element = LaidOut(decl.element, LayoutStructured, false);
            Val result{};
            result.kind = VK::Ptr;
            result.type = element;
            result.storage = spv::StorageBuffer;
            result.readOnly = decl.kind == Res::StructuredBuffer;
            result.nonUniform = object.nonUniform;
            result.id = Op(spv::OpAccessChain, PointerTo(spv::StorageBuffer, element),
                           {block.id, ConstUint(0), index.id});
            if (object.nonUniform)
            {
                Decorate(result.id, spv::DecNonUniform);
            }
            return result;
        }
        // Textures: t[coord] reads mip 0; u[coord] is a texel lvalue.
        const ImageShape shape = ShapeOf(decl.kind);
        if (shape.ms || shape.dim == 3)
        {
            (void)Fail(at, "this texture type cannot be indexed");
            return Val{};
        }
        const Val image = Load(ResourceElement(object, at));
        if (m_error)
        {
            return Val{};
        }
        const TypeId coordType = VectorOf(IntType(), shape.coordComponents);
        const Val coord = Convert(index, coordType);
        if (m_error)
        {
            return Val{};
        }
        if (shape.storage)
        {
            Val result{};
            result.kind = VK::Texel;
            result.id = image.id;
            result.aux = coord.id;
            result.type = decl.element;
            result.resource = object.resource;
            // The read or write capability is declared where the texel is
            // loaded or stored, so a write-only image needs only the write one.
            return result;
        }
        const TypeId scalar = ScalarOf(decl.element);
        const TypeId texel = VectorOf(scalar, 4);
        Val fetched{};
        fetched.kind = VK::R;
        fetched.type = texel;
        fetched.id = Op(spv::OpImageFetch, texel, {image.id, coord.id, 0x2, ConstInt(0)});
        return Convert(fetched, decl.element);
    }
    if (object.kind == VK::Ptr)
    {
        const TypeEntry entry = T(object.type);
        TypeId element = 0;
        switch (entry.kind)
        {
        case TK::Array:
        case TK::RtArray:
        case TK::Vector:
        case TK::Matrix:
            element = entry.a;
            break;
        default:
            (void)Fail(at, "this value cannot be indexed");
            return Val{};
        }
        if (!IsInteger(index.type))
        {
            index = Convert(index, UintType());
        }
        if (!IsScalar(index.type))
        {
            (void)Fail(at, "an index must be a scalar");
            return Val{};
        }
        Val result = object;
        result.type = element;
        result.id =
            Op(spv::OpAccessChain, PointerTo(object.storage, element), {object.id, index.id});
        return result;
    }
    const Val loaded = Rvalue(object);
    if (m_error)
    {
        return Val{};
    }
    const TypeEntry entry = T(loaded.type);
    if (!IsInteger(index.type))
    {
        index = Convert(index, UintType());
    }
    if (entry.kind == TK::Vector && !index.isConst)
    {
        Val result{};
        result.kind = VK::R;
        result.type = entry.a;
        result.id = Op(spv::OpVectorExtractDynamic, entry.a, {loaded.id, index.id});
        return result;
    }
    if (entry.kind == TK::Vector || entry.kind == TK::Matrix || entry.kind == TK::Array)
    {
        if (index.isConst)
        {
            if (index.constBits >= entry.b)
            {
                (void)Fail(at, "constant index out of range");
                return Val{};
            }
            Val result{};
            result.kind = VK::R;
            result.type = entry.a;
            result.id = Op(spv::OpCompositeExtract, entry.a, {loaded.id, index.constBits});
            return result;
        }
        // A dynamic index into a value: spill it to a local.
        const u32 local = NewLocal(loaded.type);
        OpVoid(spv::OpStore, {local, loaded.id});
        Val pointer{};
        pointer.kind = VK::Ptr;
        pointer.id = local;
        pointer.type = loaded.type;
        pointer.storage = spv::Function;
        pointer.readOnly = true;
        return Index(pointer, index, at);
    }
    (void)Fail(at, "this value cannot be indexed");
    return Val{};
}

// ---- Resources ------------------------------------------------------------------

bool Compiler::BindResource(u32 declIndex, const Token& at) noexcept
{
    ResourceDecl& decl = m_resources[declIndex];
    if (decl.resolved)
    {
        return true;
    }
    if (!decl.hasRegister)
    {
        // FXC assigns registers to unannotated resources; do the same, after every
        // explicit register of the class in space 0.
        u32 next = 0;
        for (u32 index = 0; index < m_resources.Size(); ++index)
        {
            const ResourceDecl& other = m_resources[index];
            if (other.registerClass == decl.registerClass && other.space == 0 && other.hasRegister)
            {
                const u32 size = other.arraySize == 0 || other.arraySize == kUnboundedArray
                                     ? 1
                                     : other.arraySize;
                next = other.registerIndex + size > next ? other.registerIndex + size : next;
            }
        }
        decl.registerIndex = next;
        decl.space = 0;
        decl.hasRegister = true;
    }

    BindingKind expected = BindingKind::UniformBuffer;
    switch (decl.kind)
    {
    case Res::CBuffer:
        expected = BindingKind::UniformBuffer;
        break;
    case Res::StructuredBuffer:
    case Res::RWStructuredBuffer:
    case Res::ByteAddressBuffer:
    case Res::RWByteAddressBuffer:
        expected = BindingKind::StorageBuffer;
        break;
    case Res::SamplerState:
    case Res::SamplerComparisonState:
        expected = BindingKind::Sampler;
        break;
    default:
        expected =
            ShapeOf(decl.kind).storage ? BindingKind::StorageImage : BindingKind::SampledImage;
        break;
    }

    ResolvedBinding binding{};
    if (m_desc.resolver != nullptr)
    {
        if (!m_desc.resolver->Resolve(decl.registerClass, decl.registerIndex, decl.space, &binding))
        {
            static const char kClass[] = {'b', 't', 'u', 's'};
            char message[160] = {};
            std::snprintf(message, sizeof(message),
                          "register %c%u, space%u has no binding in the pipeline layout",
                          kClass[static_cast<u32>(decl.registerClass)],
                          static_cast<unsigned>(decl.registerIndex),
                          static_cast<unsigned>(decl.space));
            // The diagnostic buffer copies the message.
            return Fail(at, message);
        }
    }
    else
    {
        // Without a layout, each register class gets its own binding range so
        // b0, t0, u0 and s0 do not collide (DXC's -fvk-*-shift convention).
        static const u32 kClassShift[] = {0, 1000, 2000, 3000};
        binding.kind = expected;
        binding.set = decl.space;
        binding.binding = decl.registerIndex + kClassShift[static_cast<u32>(decl.registerClass)];
        binding.arrayCount = decl.arraySize == 0                 ? 1
                             : decl.arraySize == kUnboundedArray ? 0
                                                                 : decl.arraySize;
        binding.firstElement = 0;
    }
    if (binding.kind != expected &&
        !(binding.kind == BindingKind::PushConstant && expected == BindingKind::UniformBuffer))
    {
        return Fail(at,
                    "the register maps to a binding of a different type in the pipeline layout");
    }
    if (binding.kind == BindingKind::PushConstant && decl.arraySize != 0)
    {
        return Fail(at, "a push-constant block cannot be an array");
    }
    if (decl.arraySize != 0 && decl.arraySize != kUnboundedArray && binding.arrayCount != 0 &&
        binding.firstElement + decl.arraySize > binding.arrayCount)
    {
        return Fail(at, "the resource array is larger than its binding in the pipeline layout");
    }
    decl.binding = binding;
    decl.resolved = true;

    // The descriptor's own type.
    TypeId descriptor = 0;
    u32 storage = spv::UniformConstant;
    switch (decl.kind)
    {
    case Res::CBuffer:
        descriptor = StructType(decl.structDef, LayoutCBuffer, true);
        storage = binding.kind == BindingKind::PushConstant ? spv::PushConstant : spv::Uniform;
        break;
    case Res::StructuredBuffer:
    case Res::RWStructuredBuffer:
    case Res::ByteAddressBuffer:
    case Res::RWByteAddressBuffer:
    {
        const bool raw =
            decl.kind == Res::ByteAddressBuffer || decl.kind == Res::RWByteAddressBuffer;
        const TypeId element = raw ? UintType() : decl.element;
        // A synthetic wrapper structure { T data[]; }.
        StructDef wrapper{};
        StructMember data{};
        data.name = Name{"data", 4};
        data.type = RuntimeArrayOf(element, 0);
        wrapper.firstMember = m_members.Size();
        wrapper.memberCount = 1;
        u32 defIndex = 0;
        for (u32 index = 1; index < m_structs.Size(); ++index)
        {
            const StructDef& existing = m_structs[index];
            if (existing.memberCount == 1 && existing.name.length == 0 &&
                m_members[existing.firstMember].type == data.type)
            {
                defIndex = index;
            }
        }
        if (defIndex == 0)
        {
            if (!m_members.Push(data) || !m_structs.Push(wrapper))
            {
                return Fail(at, "out of memory");
            }
            defIndex = m_structs.Size() - 1;
        }
        descriptor = StructType(defIndex, LayoutStructured, true);
        storage = spv::StorageBuffer;
        break;
    }
    case Res::SamplerState:
    case Res::SamplerComparisonState:
        descriptor = Intern(TK::Sampler);
        break;
    default:
    {
        const ImageShape shape = ShapeOf(decl.kind);
        const TypeId scalar = ScalarOf(decl.element);
        const u32 sampled = shape.storage ? 2U : 1U;
        const u32 depth = shape.storage ? 0U : 2U;
        const u32 flags =
            (shape.arrayed ? 1U : 0U) | (shape.ms ? 2U : 0U) | (sampled << 2) | (depth << 4);
        descriptor = Intern(TK::Image, scalar, shape.dim, flags);
        if (shape.dim == 0)
        {
            Capability(shape.storage ? spv::CapImage1D : spv::CapSampled1D);
        }
        if (shape.dim == 3 && shape.arrayed)
        {
            Capability(spv::CapSampledCubeArray);
        }
        break;
    }
    }
    decl.blockType = descriptor;
    decl.storage = storage;

    // One variable per (set, binding): every declaration landing there shares it.
    for (u32 index = 0; index < m_bindingVars.Size(); ++index)
    {
        const BindingVar& existing = m_bindingVars[index];
        if (binding.kind != BindingKind::PushConstant && existing.set == binding.set &&
            existing.binding == binding.binding && existing.variable != 0)
        {
            const TypeId existingElement =
                T(existing.pointee).kind == TK::Array || T(existing.pointee).kind == TK::RtArray
                    ? T(existing.pointee).a
                    : existing.pointee;
            if (existingElement != descriptor)
            {
                return Fail(at, "two resources of different types share one binding");
            }
            decl.variable = existing.variable;
            decl.variableType = existing.pointee;
            decl.binds = existingElement != existing.pointee;
            return true;
        }
    }
    TypeId pointee = descriptor;
    const bool arrayed = binding.arrayCount != 1 || decl.arraySize != 0;
    if (arrayed)
    {
        if (binding.kind == BindingKind::PushConstant)
        {
            return Fail(at, "a push-constant block cannot be an array");
        }
        if (binding.arrayCount == 0)
        {
            Capability(spv::CapRuntimeDescriptorArray);
            pointee = RuntimeArrayOf(descriptor, 0);
        }
        else
        {
            pointee = ArrayOf(descriptor, binding.arrayCount, 0);
        }
    }
    const u32 variable = NewId();
    Inst(m_globals, spv::OpVariable, {SpirvType(PointerTo(storage, pointee)), variable, storage});
    (void)m_interface.Push(variable);
    if (binding.kind != BindingKind::PushConstant)
    {
        Decorate(variable, spv::DecDescriptorSet, {binding.set});
        Decorate(variable, spv::DecBinding, {binding.binding});
    }
    if (decl.kind == Res::StructuredBuffer || decl.kind == Res::ByteAddressBuffer)
    {
        Decorate(variable, spv::DecNonWritable);
    }
    decl.variable = variable;
    decl.variableType = pointee;
    decl.binds = arrayed;
    BindingVar record{};
    record.set = binding.kind == BindingKind::PushConstant ? 0xFFFFFFFFU : binding.set;
    record.binding = binding.binding;
    record.variable = variable;
    record.pointee = pointee;
    (void)m_bindingVars.Push(record);
    return true;
}

Val Compiler::ResourceElement(const Val& resource, const Token& at) noexcept
{
    const u32 declIndex = resource.resource - 1;
    if (!BindResource(declIndex, at))
    {
        return Val{};
    }
    const ResourceDecl& decl = m_resources[declIndex];
    if (decl.arraySize != 0 && resource.aux == 0)
    {
        (void)Fail(at, "a resource array must be indexed");
        return Val{};
    }
    Val result{};
    result.kind = VK::Ptr;
    result.type = decl.blockType;
    result.storage = decl.storage;
    result.readOnly = true;
    result.nonUniform = resource.nonUniform;
    if (!decl.binds)
    {
        result.id = decl.variable;
        return result;
    }
    u32 element = 0;
    const u32 first = decl.binding.firstElement;
    if (resource.aux != 0)
    {
        element = first == 0 ? resource.aux
                             : Op(spv::OpIAdd, UintType(), {resource.aux, ConstUint(first)});
    }
    else
    {
        element = ConstUint(first);
    }
    result.id =
        Op(spv::OpAccessChain, PointerTo(decl.storage, decl.blockType), {decl.variable, element});
    if (resource.nonUniform)
    {
        Capability(spv::CapShaderNonUniform);
        switch (decl.kind)
        {
        case Res::CBuffer:
            Capability(spv::CapUniformBufferArrayNonUniformIndexing);
            break;
        case Res::StructuredBuffer:
        case Res::RWStructuredBuffer:
        case Res::ByteAddressBuffer:
        case Res::RWByteAddressBuffer:
            Capability(spv::CapStorageBufferArrayNonUniformIndexing);
            break;
        case Res::SamplerState:
        case Res::SamplerComparisonState:
            Capability(spv::CapSampledImageArrayNonUniformIndexing);
            break;
        default:
            Capability(ShapeOf(decl.kind).storage ? spv::CapStorageImageArrayNonUniformIndexing
                                                  : spv::CapSampledImageArrayNonUniformIndexing);
            break;
        }
        Decorate(result.id, spv::DecNonUniform);
    }
    return result;
}

Val Compiler::CBufferMember(u32 declIndex, u32 member, const Token& at) noexcept
{
    Val resource{};
    resource.kind = VK::Resource;
    resource.resource = declIndex + 1;
    const Val block = ResourceElement(resource, at);
    if (m_error)
    {
        return Val{};
    }
    const ResourceDecl& decl = m_resources[declIndex];
    const StructDef& def = m_structs[decl.structDef];
    const TypeId memberType =
        LaidOut(m_members[def.firstMember + member].type, LayoutCBuffer, false);
    Val result{};
    result.kind = VK::Ptr;
    result.type = memberType;
    result.storage = decl.storage;
    result.readOnly = true;
    result.id =
        Op(spv::OpAccessChain, PointerTo(decl.storage, memberType), {block.id, ConstUint(member)});
    return result;
}

// ---- Calls ----------------------------------------------------------------------

Val Compiler::CallFunction(const Name& name, Val* args, u32 count, const Token& at) noexcept
{
    // Pick the overload whose parameter types match best: exact matches first.
    u32 best = 0xFFFFFFFFU;
    int bestScore = -1;
    for (u32 index = 0; index < m_funcs.Size(); ++index)
    {
        const FuncDecl& decl = m_funcs[index];
        if (!decl.name.Equals(name) || decl.paramCount != count)
        {
            continue;
        }
        int score = 1;
        for (u32 param = 0; param < count; ++param)
        {
            const TypeId want = m_params[decl.firstParam + param].type;
            const TypeId have = args[param].kind == VK::Swz
                                    ? VectorOf(ScalarOf(args[param].type), args[param].swizzleCount)
                                    : LogicalOf(args[param].type);
            score += want == have ? 4 : (ComponentCount(want) == ComponentCount(have) ? 1 : 0);
        }
        // Prefer a definition over a prototype with the same signature.
        if (score > bestScore || (score == bestScore && decl.bodyToken != 0))
        {
            best = index;
            bestScore = score;
        }
    }
    if (best == 0xFFFFFFFFU)
    {
        (void)Fail(at, "no overload of this function takes these arguments");
        return Val{};
    }
    // A prototype resolves to the definition with the same signature.
    if (m_funcs[best].bodyToken == 0)
    {
        for (u32 index = 0; index < m_funcs.Size(); ++index)
        {
            const FuncDecl& decl = m_funcs[index];
            if (decl.bodyToken == 0 || !decl.name.Equals(name) || decl.paramCount != count)
            {
                continue;
            }
            bool same = true;
            for (u32 param = 0; param < count && same; ++param)
            {
                same = m_params[decl.firstParam + param].type ==
                       m_params[m_funcs[best].firstParam + param].type;
            }
            if (same)
            {
                best = index;
            }
        }
    }
    u32 operands[33] = {};
    u32 temps[32] = {};
    for (u32 param = 0; param < count; ++param)
    {
        const Param info = m_params[m_funcs[best].firstParam + param];
        if (info.dir == ParamDir::In)
        {
            operands[param + 1] = Convert(args[param], info.type).id;
        }
        else
        {
            if (args[param].kind != VK::Ptr && args[param].kind != VK::Swz &&
                args[param].kind != VK::Texel)
            {
                (void)Fail(at, "an out or inout argument must be assignable");
                return Val{};
            }
            temps[param] = NewLocal(info.type);
            if (info.dir == ParamDir::InOut)
            {
                const Val current = Convert(args[param], info.type);
                OpVoid(spv::OpStore, {temps[param], current.id});
            }
            operands[param + 1] = temps[param];
        }
        if (m_error)
        {
            return Val{};
        }
    }
    if (!CompileFunction(best))
    {
        return Val{};
    }
    const FuncDecl& decl = m_funcs[best];
    operands[0] = decl.spirvId;
    Val result{};
    result.kind = VK::R;
    result.type = decl.returnType;
    result.id = OpN(spv::OpFunctionCall, decl.returnType, operands, count + 1);
    for (u32 param = 0; param < count; ++param)
    {
        if (temps[param] == 0)
        {
            continue;
        }
        const Param info = m_params[decl.firstParam + param];
        Val value{};
        value.kind = VK::R;
        value.type = info.type;
        value.id = Op(spv::OpLoad, info.type, {temps[param]});
        if (!Store(args[param], value))
        {
            return Val{};
        }
    }
    if (T(decl.returnType).kind == TK::Void)
    {
        result.kind = VK::None;
    }
    return result;
}

Val Compiler::AtomicOnPointer(u32 opcode, const Val& pointer, Val value, Val comparator,
                              const Token& at) noexcept
{
    if (pointer.kind != VK::Ptr || pointer.readOnly ||
        (pointer.storage != spv::StorageBuffer && pointer.storage != spv::Workgroup))
    {
        (void)Fail(at, "an interlocked destination must be a writable buffer element or "
                       "groupshared variable");
        return Val{};
    }
    if (!IsScalar(pointer.type) || !IsInteger(pointer.type))
    {
        (void)Fail(at, "interlocked operations need a 32-bit integer destination");
        return Val{};
    }
    u32 actual = opcode;
    const bool isSigned = IsSigned(pointer.type);
    if (opcode == spv::OpAtomicUMin && isSigned)
    {
        actual = spv::OpAtomicSMin;
    }
    if (opcode == spv::OpAtomicUMax && isSigned)
    {
        actual = spv::OpAtomicSMax;
    }
    const u32 scope = ConstUint(pointer.storage == spv::Workgroup ? 2U : 1U);
    const u32 semantics = ConstUint(0);
    const Val converted = Convert(value, pointer.type);
    if (m_error)
    {
        return Val{};
    }
    Val result{};
    result.kind = VK::R;
    result.type = pointer.type;
    if (actual == spv::OpAtomicCompareExchange)
    {
        const Val compare = Convert(comparator, pointer.type);
        result.id = Op(actual, pointer.type,
                       {pointer.id, scope, semantics, semantics, converted.id, compare.id});
    }
    else
    {
        result.id = Op(actual, pointer.type, {pointer.id, scope, semantics, converted.id});
    }
    return result;
}

Val Compiler::Atomic(const char* name, const Val& dest, Val* args, u32 count,
                     const Token& at) noexcept
{
    // `args` are the operands after the destination.
    u32 opcode = 0;
    if (Same(name, "InterlockedAdd"))
        opcode = spv::OpAtomicIAdd;
    else if (Same(name, "InterlockedMin"))
        opcode = spv::OpAtomicUMin;
    else if (Same(name, "InterlockedMax"))
        opcode = spv::OpAtomicUMax;
    else if (Same(name, "InterlockedAnd"))
        opcode = spv::OpAtomicAnd;
    else if (Same(name, "InterlockedOr"))
        opcode = spv::OpAtomicOr;
    else if (Same(name, "InterlockedXor"))
        opcode = spv::OpAtomicXor;
    else if (Same(name, "InterlockedExchange"))
        opcode = spv::OpAtomicExchange;
    else if (Same(name, "InterlockedCompareExchange") || Same(name, "InterlockedCompareStore"))
        opcode = spv::OpAtomicCompareExchange;
    else
    {
        (void)Fail(at, "unknown interlocked operation");
        return Val{};
    }
    if (opcode == spv::OpAtomicCompareExchange)
    {
        const bool store = Same(name, "InterlockedCompareStore");
        if (count != (store ? 2U : 3U))
        {
            (void)Fail(at, "wrong number of arguments to an interlocked compare");
            return Val{};
        }
        const Val original = AtomicOnPointer(opcode, dest, args[1], args[0], at);
        if (!store && !Store(args[2], original))
        {
            return Val{};
        }
        Val none{};
        return none;
    }
    if (count < 1 || count > 2 || (opcode == spv::OpAtomicExchange && count != 2))
    {
        (void)Fail(at, "wrong number of arguments to an interlocked operation");
        return Val{};
    }
    const Val original = AtomicOnPointer(opcode, dest, args[0], Val{}, at);
    if (count == 2 && !Store(args[1], original))
    {
        return Val{};
    }
    return Val{};
}

namespace
{

struct Unary
{
    const char* name;
    u32 glsl;
};

constexpr Unary kFloatUnary[] = {
    {"sqrt", spv::GlslSqrt},       {"rsqrt", spv::GlslInverseSqrt}, {"exp", spv::GlslExp},
    {"exp2", spv::GlslExp2},       {"log", spv::GlslLog},           {"log2", spv::GlslLog2},
    {"sin", spv::GlslSin},         {"cos", spv::GlslCos},           {"tan", spv::GlslTan},
    {"asin", spv::GlslAsin},       {"acos", spv::GlslAcos},         {"atan", spv::GlslAtan},
    {"sinh", spv::GlslSinh},       {"cosh", spv::GlslCosh},         {"tanh", spv::GlslTanh},
    {"floor", spv::GlslFloor},     {"ceil", spv::GlslCeil},         {"frac", spv::GlslFract},
    {"trunc", spv::GlslTrunc},     {"round", spv::GlslRoundEven},   {"radians", spv::GlslRadians},
    {"degrees", spv::GlslDegrees},
};

}  // namespace

Val Compiler::CallIntrinsic(const Name& name, Val* args, u32 count, const Token& at,
                            bool* handled) noexcept
{
    *handled = true;
    char text[40] = {};
    if (name.length >= sizeof(text))
    {
        *handled = false;
        return Val{};
    }
    std::memcpy(text, name.text, name.length);
    const auto argCount = [&](u32 wanted) noexcept
    {
        if (count != wanted)
        {
            (void)Fail(at, "wrong number of arguments to an intrinsic");
            return false;
        }
        return true;
    };
    const auto rvalues = [&]() noexcept
    {
        for (u32 index = 0; index < count; ++index)
        {
            args[index] = Rvalue(args[index]);
        }
        return !m_error;
    };
    Val result{};
    result.kind = VK::R;

    // Barriers.
    if (Same(text, "GroupMemoryBarrierWithGroupSync") || Same(text, "GroupMemoryBarrier") ||
        Same(text, "DeviceMemoryBarrierWithGroupSync") || Same(text, "DeviceMemoryBarrier") ||
        Same(text, "AllMemoryBarrierWithGroupSync") || Same(text, "AllMemoryBarrier"))
    {
        if (!argCount(0))
        {
            return Val{};
        }
        if (m_desc.stage != ShaderStage::Compute && std::strstr(text, "WithGroupSync") != nullptr)
        {
            (void)Fail(at, "group sync is only valid in a compute shader");
            return Val{};
        }
        const bool group = text[0] == 'G';
        const bool all = text[0] == 'A';
        const u32 memoryScope = ConstUint(group ? 2U : 1U);
        const u32 semantics = ConstUint(group ? 0x108U : all ? 0x948U : 0x848U);
        if (std::strstr(text, "WithGroupSync") != nullptr)
        {
            OpVoid(spv::OpControlBarrier, {ConstUint(2), memoryScope, semantics});
        }
        else
        {
            OpVoid(spv::OpMemoryBarrier, {memoryScope, semantics});
        }
        return Val{};
    }
    if (std::strncmp(text, "Interlocked", 11) == 0)
    {
        if (count < 2)
        {
            (void)Fail(at, "wrong number of arguments to an interlocked operation");
            return Val{};
        }
        return Atomic(text, args[0], args + 1, count - 1, at);
    }
    if (Same(text, "NonUniformResourceIndex"))
    {
        if (!argCount(1) || !rvalues())
        {
            return Val{};
        }
        Val value = args[0];
        value.nonUniform = true;
        return value;
    }
    if (Same(text, "clip"))
    {
        if (!argCount(1) || !rvalues())
        {
            return Val{};
        }
        if (m_desc.stage != ShaderStage::Pixel)
        {
            (void)Fail(at, "clip is only valid in a pixel shader");
            return Val{};
        }
        const Val zero = ConstVal(FloatType(), 0);
        Val negative = Binary("<", args[0], zero, at);
        if (ComponentCount(negative.type) > 1)
        {
            const u32 any = Op(spv::OpAny, BoolType(), {negative.id});
            negative = Val{};
            negative.kind = VK::R;
            negative.type = BoolType();
            negative.id = any;
        }
        const u32 killLabel = NewId();
        const u32 merge = NewId();
        OpVoid(spv::OpSelectionMerge, {merge, 0});
        Terminate(spv::OpBranchConditional, {negative.id, killLabel, merge});
        Label(killLabel);
        Terminate(spv::OpKill);
        Label(merge);
        return Val{};
    }

    for (const Unary& entry : kFloatUnary)
    {
        if (Same(text, entry.name))
        {
            if (!argCount(1) || !rvalues())
            {
                return Val{};
            }
            const TypeId type = WithScalar(args[0].type, FloatType());
            const Val x = Convert(args[0], type);
            if (m_error)
            {
                return Val{};
            }
            result.type = type;
            result.id = Ext(entry.glsl, type, {x.id});
            return result;
        }
    }
    if (Same(text, "log10"))
    {
        if (!argCount(1) || !rvalues())
        {
            return Val{};
        }
        const TypeId type = WithScalar(args[0].type, FloatType());
        Val log2{};
        log2.kind = VK::R;
        log2.type = type;
        log2.id = Ext(spv::GlslLog2, type, {Convert(args[0], type).id});
        return Arithmetic("*", log2, ConstVal(FloatType(), FloatBits(0.30102999566F)), at);
    }
    if (Same(text, "saturate"))
    {
        if (!argCount(1) || !rvalues())
        {
            return Val{};
        }
        const TypeId type = WithScalar(args[0].type, FloatType());
        const Val x = Convert(args[0], type);
        const Val zero = Splat(ConstVal(FloatType(), 0), type);
        const Val one = Splat(ConstVal(FloatType(), FloatBits(1.0F)), type);
        result.type = type;
        result.id = Ext(spv::GlslFClamp, type, {x.id, zero.id, one.id});
        return result;
    }
    if (Same(text, "abs") || Same(text, "sign"))
    {
        if (!argCount(1) || !rvalues())
        {
            return Val{};
        }
        TypeId type = args[0].type;
        if (IsBoolish(type))
        {
            type = WithScalar(type, IntType());
        }
        const Val x = Convert(args[0], type);
        const bool isAbs = Same(text, "abs");
        const TK kind = T(ScalarOf(type)).kind;
        if (kind == TK::Uint)
        {
            if (isAbs)
            {
                return x;
            }
            // sign(uint) is 0 or 1.
            const Val positive = Binary(">", x, ConstVal(UintType(), 0), at);
            return Convert(positive, WithScalar(type, IntType()));
        }
        result.type = type;
        result.id = Ext(kind == TK::Float ? (isAbs ? spv::GlslFAbs : spv::GlslFSign)
                                          : (isAbs ? spv::GlslSAbs : spv::GlslSSign),
                        type, {x.id});
        if (!isAbs)
        {
            return Convert(result, WithScalar(type, IntType()));  // HLSL sign returns int
        }
        return result;
    }
    if (Same(text, "min") || Same(text, "max") || Same(text, "clamp"))
    {
        const bool isClamp = Same(text, "clamp");
        if (!argCount(isClamp ? 3 : 2) || !rvalues())
        {
            return Val{};
        }
        TypeId type = CommonType(args, count, true);
        if (type == 0)
        {
            (void)Fail(at, "invalid arguments to min, max or clamp");
            return Val{};
        }
        if (IsBoolish(type))
        {
            type = WithScalar(type, IntType());
        }
        u32 ids[3] = {};
        bool allConst = true;
        u32 bits[3] = {};
        for (u32 index = 0; index < count; ++index)
        {
            const Val converted = Convert(args[index], type);
            ids[index] = converted.id;
            allConst = allConst && converted.isConst && IsScalar(type);
            bits[index] = converted.constBits;
        }
        if (m_error)
        {
            return Val{};
        }
        const TK kind = T(ScalarOf(type)).kind;
        if (allConst && !isClamp)
        {
            bool aLess = false;
            if (kind == TK::Float)
                aLess = BitsFloat(bits[0]) < BitsFloat(bits[1]);
            else if (kind == TK::Int)
                aLess = static_cast<int>(bits[0]) < static_cast<int>(bits[1]);
            else
                aLess = bits[0] < bits[1];
            const bool takeA = Same(text, "min") ? aLess : !aLess;
            return ConstVal(type, takeA ? bits[0] : bits[1]);
        }
        u32 glsl = 0;
        if (isClamp)
            glsl = kind == TK::Float ? spv::GlslFClamp
                   : kind == TK::Int ? spv::GlslSClamp
                                     : spv::GlslUClamp;
        else if (Same(text, "min"))
            glsl = kind == TK::Float ? spv::GlslFMin
                   : kind == TK::Int ? spv::GlslSMin
                                     : spv::GlslUMin;
        else
            glsl = kind == TK::Float ? spv::GlslFMax
                   : kind == TK::Int ? spv::GlslSMax
                                     : spv::GlslUMax;
        result.type = type;
        result.id =
            isClamp ? Ext(glsl, type, {ids[0], ids[1], ids[2]}) : Ext(glsl, type, {ids[0], ids[1]});
        return result;
    }
    // Float functions of several same-shaped arguments.
    struct Multi
    {
        const char* name;
        u32 glsl;
        u32 args;
    };
    static constexpr Multi kFloatMulti[] = {
        {"lerp", spv::GlslFMix, 3},
        {"pow", spv::GlslPow, 2},
        {"atan2", spv::GlslAtan2, 2},
        {"step", spv::GlslStep, 2},
        {"smoothstep", spv::GlslSmoothStep, 3},
        {"reflect", spv::GlslReflect, 2},
        {"faceforward", spv::GlslFaceForward, 3},
        {"ldexp", spv::GlslLdexp, 2},
    };
    for (const Multi& entry : kFloatMulti)
    {
        if (!Same(text, entry.name))
        {
            continue;
        }
        if (!argCount(entry.args) || !rvalues())
        {
            return Val{};
        }
        TypeId type = CommonType(args, count, true);
        if (type == 0)
        {
            (void)Fail(at, "invalid intrinsic arguments");
            return Val{};
        }
        type = WithScalar(type, FloatType());
        u32 ids[3] = {};
        for (u32 index = 0; index < count; ++index)
        {
            ids[index] = Convert(args[index], type).id;
        }
        if (m_error)
        {
            return Val{};
        }
        if (entry.glsl == spv::GlslLdexp)
        {
            // ldexp takes an integer exponent in SPIR-V.
            ids[1] = Convert(args[1], WithScalar(type, IntType())).id;
        }
        result.type = type;
        result.id = entry.args == 3 ? Ext(entry.glsl, type, {ids[0], ids[1], ids[2]})
                                    : Ext(entry.glsl, type, {ids[0], ids[1]});
        return result;
    }
    if (Same(text, "fmod"))
    {
        if (!argCount(2) || !rvalues())
        {
            return Val{};
        }
        const TypeId type = WithScalar(CommonType(args, 2, true), FloatType());
        const Val a = Convert(args[0], type);
        const Val b = Convert(args[1], type);
        result.type = type;
        result.id = Op(spv::OpFRem, type, {a.id, b.id});
        return result;
    }
    if (Same(text, "mad"))
    {
        if (!argCount(3) || !rvalues())
        {
            return Val{};
        }
        return Arithmetic("+", Arithmetic("*", args[0], args[1], at), args[2], at);
    }
    if (Same(text, "refract"))
    {
        if (!argCount(3) || !rvalues())
        {
            return Val{};
        }
        const TypeId type = WithScalar(CommonType(args, 2, true), FloatType());
        const Val i = Convert(args[0], type);
        const Val n = Convert(args[1], type);
        const Val eta = Convert(args[2], FloatType());
        result.type = type;
        result.id = Ext(spv::GlslRefract, type, {i.id, n.id, eta.id});
        return result;
    }
    if (Same(text, "dot") || Same(text, "distance") || Same(text, "cross") ||
        Same(text, "length") || Same(text, "normalize"))
    {
        const bool unary = Same(text, "length") || Same(text, "normalize");
        if (!argCount(unary ? 1 : 2) || !rvalues())
        {
            return Val{};
        }
        TypeId type = CommonType(args, count, true);
        if (type == 0)
        {
            (void)Fail(at, "invalid intrinsic arguments");
            return Val{};
        }
        if (Same(text, "dot") && IsInteger(type) && !IsScalar(type))
        {
            // Integer dot: multiply and add the components.
            const Val products =
                Arithmetic("*", Convert(args[0], type), Convert(args[1], type), at);
            const TypeId scalar = ScalarOf(type);
            Val sum{};
            for (u32 component = 0; component < ComponentCount(type); ++component)
            {
                Val part{};
                part.kind = VK::R;
                part.type = scalar;
                part.id = Op(spv::OpCompositeExtract, scalar, {products.id, component});
                sum = component == 0 ? part : Arithmetic("+", sum, part, at);
            }
            return sum;
        }
        type = WithScalar(type, FloatType());
        const Val a = Convert(args[0], type);
        if (IsScalar(type))
        {
            if (Same(text, "dot"))
                return Arithmetic("*", a, Convert(args[1], type), at);
            if (Same(text, "length"))
            {
                result.type = type;
                result.id = Ext(spv::GlslFAbs, type, {a.id});
                return result;
            }
            if (Same(text, "distance"))
            {
                const Val difference = Arithmetic("-", a, Convert(args[1], type), at);
                result.type = type;
                result.id = Ext(spv::GlslFAbs, type, {difference.id});
                return result;
            }
            if (Same(text, "normalize"))
            {
                result.type = type;
                result.id = Ext(spv::GlslFSign, type, {a.id});
                return result;
            }
            (void)Fail(at, "cross needs float3 arguments");
            return Val{};
        }
        if (Same(text, "dot"))
        {
            const Val b = Convert(args[1], type);
            result.type = FloatType();
            result.id = Op(spv::OpDot, FloatType(), {a.id, b.id});
            if (m_precise)
            {
                Decorate(result.id, spv::DecNoContraction);
            }
            return result;
        }
        if (Same(text, "length"))
        {
            result.type = FloatType();
            result.id = Ext(spv::GlslLength, FloatType(), {a.id});
            return result;
        }
        if (Same(text, "normalize"))
        {
            result.type = type;
            result.id = Ext(spv::GlslNormalize, type, {a.id});
            return result;
        }
        const Val b = Convert(args[1], type);
        if (Same(text, "distance"))
        {
            result.type = FloatType();
            result.id = Ext(spv::GlslDistance, FloatType(), {a.id, b.id});
            return result;
        }
        if (ComponentCount(type) != 3)
        {
            (void)Fail(at, "cross needs float3 arguments");
            return Val{};
        }
        result.type = type;
        result.id = Ext(spv::GlslCross, type, {a.id, b.id});
        return result;
    }
    if (Same(text, "isnan") || Same(text, "isinf") || Same(text, "isfinite"))
    {
        if (!argCount(1) || !rvalues())
        {
            return Val{};
        }
        const TypeId type = WithScalar(args[0].type, FloatType());
        const TypeId boolType = WithScalar(type, BoolType());
        const Val x = Convert(args[0], type);
        if (Same(text, "isfinite"))
        {
            const u32 nan = Op(spv::OpIsNan, boolType, {x.id});
            const u32 inf = Op(spv::OpIsInf, boolType, {x.id});
            const u32 either = Op(spv::OpLogicalOr, boolType, {nan, inf});
            result.type = boolType;
            result.id = Op(spv::OpLogicalNot, boolType, {either});
            return result;
        }
        result.type = boolType;
        result.id = Op(Same(text, "isnan") ? spv::OpIsNan : spv::OpIsInf, boolType, {x.id});
        return result;
    }
    if (Same(text, "any") || Same(text, "all"))
    {
        if (!argCount(1) || !rvalues())
        {
            return Val{};
        }
        const Val b = ToBool(args[0]);
        if (IsScalar(b.type))
        {
            return b;
        }
        result.type = BoolType();
        result.id = Op(Same(text, "any") ? spv::OpAny : spv::OpAll, BoolType(), {b.id});
        return result;
    }
    if (Same(text, "asuint") || Same(text, "asint") || Same(text, "asfloat"))
    {
        if (!argCount(1) || !rvalues())
        {
            return Val{};
        }
        if (!IsNumericScalarOrVector(args[0].type) || IsBoolish(args[0].type))
        {
            (void)Fail(at, "invalid argument to a bit cast");
            return Val{};
        }
        const TypeId scalar = Same(text, "asuint")  ? UintType()
                              : Same(text, "asint") ? IntType()
                                                    : FloatType();
        const TypeId type = WithScalar(args[0].type, scalar);
        if (type == args[0].type)
        {
            return args[0];
        }
        if (args[0].isConst && IsScalar(type))
        {
            return ConstVal(type, args[0].constBits);
        }
        result.type = type;
        result.id = Op(spv::OpBitcast, type, {args[0].id});
        return result;
    }
    if (Same(text, "countbits") || Same(text, "reversebits") || Same(text, "firstbithigh") ||
        Same(text, "firstbitlow"))
    {
        if (!argCount(1) || !rvalues())
        {
            return Val{};
        }
        TypeId type = args[0].type;
        if (!IsInteger(type))
        {
            type = WithScalar(type, UintType());
        }
        const Val x = Convert(args[0], type);
        result.type = type;
        if (Same(text, "countbits"))
            result.id = Op(spv::OpBitCount, type, {x.id});
        else if (Same(text, "reversebits"))
            result.id = Op(spv::OpBitReverse, type, {x.id});
        else if (Same(text, "firstbitlow"))
            result.id = Ext(spv::GlslFindILsb, type, {x.id});
        else
            result.id = Ext(IsSigned(type) ? spv::GlslFindSMsb : spv::GlslFindUMsb, type, {x.id});
        return Convert(result, WithScalar(type, UintType()));
    }
    if (Same(text, "ddx") || Same(text, "ddy") || Same(text, "fwidth") ||
        Same(text, "ddx_coarse") || Same(text, "ddy_coarse") || Same(text, "ddx_fine") ||
        Same(text, "ddy_fine"))
    {
        if (!argCount(1) || !rvalues())
        {
            return Val{};
        }
        if (m_desc.stage != ShaderStage::Pixel)
        {
            (void)Fail(at, "derivatives are only valid in a pixel shader");
            return Val{};
        }
        const TypeId type = WithScalar(args[0].type, FloatType());
        const Val x = Convert(args[0], type);
        result.type = type;
        const u32 opcode = text[0] == 'f'   ? spv::OpFwidth
                           : text[2] == 'x' ? spv::OpDPdx
                                            : spv::OpDPdy;
        result.id = Op(opcode, type, {x.id});
        return result;
    }
    if (Same(text, "f16tof32") || Same(text, "f32tof16"))
    {
        if (!argCount(1) || !rvalues())
        {
            return Val{};
        }
        if (!IsScalar(args[0].type))
        {
            (void)Fail(at, "only the scalar forms of f16tof32 and f32tof16 are supported");
            return Val{};
        }
        const TypeId float2 = VectorOf(FloatType(), 2);
        if (Same(text, "f16tof32"))
        {
            const Val bits = Convert(args[0], UintType());
            const u32 pair = Ext(spv::GlslUnpackHalf2x16, float2, {bits.id});
            result.type = FloatType();
            result.id = Op(spv::OpCompositeExtract, FloatType(), {pair, 0});
            return result;
        }
        const Val x = Convert(args[0], FloatType());
        u32 parts[2] = {x.id, ConstFloat(0.0F)};
        const u32 pair = OpN(spv::OpCompositeConstruct, float2, parts, 2);
        result.type = UintType();
        result.id = Ext(spv::GlslPackHalf2x16, UintType(), {pair});
        return result;
    }
    if (Same(text, "transpose") || Same(text, "determinant"))
    {
        if (!argCount(1) || !rvalues())
        {
            return Val{};
        }
        const TypeEntry m = T(args[0].type);
        if (m.kind != TK::Matrix)
        {
            (void)Fail(at, "expected a matrix");
            return Val{};
        }
        if (Same(text, "determinant"))
        {
            if (m.b != T(m.a).b)
            {
                (void)Fail(at, "determinant needs a square matrix");
                return Val{};
            }
            result.type = FloatType();
            result.id = Ext(spv::GlslDeterminant, FloatType(), {args[0].id});
            return result;
        }
        const TypeId transposed = MatrixOf(FloatType(), T(m.a).b, m.b);
        result.type = transposed;
        result.id = Op(spv::OpTranspose, transposed, {args[0].id});
        return result;
    }
    if (Same(text, "mul"))
    {
        if (!argCount(2) || !rvalues())
        {
            return Val{};
        }
        const Val a = args[0];
        const Val b = args[1];
        const bool am = T(a.type).kind == TK::Matrix;
        const bool bm = T(b.type).kind == TK::Matrix;
        if (!am && !bm)
        {
            if (IsScalar(a.type) || IsScalar(b.type))
            {
                return Arithmetic("*", a, b, at);
            }
            Val pair[2] = {a, b};
            bool inner = false;
            return CallIntrinsic(Name{"dot", 3}, pair, 2, at, &inner);
        }
        if (am && bm)
        {
            const u32 aRows = T(a.type).b;
            const u32 aColumns = T(T(a.type).a).b;
            const u32 bRows = T(b.type).b;
            const u32 bColumns = T(T(b.type).a).b;
            if (aColumns != bRows)
            {
                (void)Fail(at, "matrix dimensions do not match in mul");
                return Val{};
            }
            const TypeId type = MatrixOf(FloatType(), aRows, bColumns);
            result.type = type;
            result.id = Op(spv::OpMatrixTimesMatrix, type, {b.id, a.id});
            return result;
        }
        if (am)
        {
            // mul(M, v): v has the matrix's column count, the result its row count.
            if (IsScalar(b.type))
            {
                return Arithmetic("*", a, b, at);
            }
            const u32 rows = T(a.type).b;
            const u32 columns = T(T(a.type).a).b;
            const Val v = Convert(b, VectorOf(FloatType(), columns));
            if (m_error)
            {
                return Val{};
            }
            const TypeId type = VectorOf(FloatType(), rows);
            result.type = type;
            result.id = Op(spv::OpVectorTimesMatrix, type, {v.id, a.id});
            if (m_precise)
            {
                Decorate(result.id, spv::DecNoContraction);
            }
            return result;
        }
        if (IsScalar(a.type))
        {
            return Arithmetic("*", b, a, at);
        }
        // mul(v, M): v has the matrix's row count, the result its column count.
        const u32 rows = T(b.type).b;
        const u32 columns = T(T(b.type).a).b;
        const Val v = Convert(a, VectorOf(FloatType(), rows));
        if (m_error)
        {
            return Val{};
        }
        const TypeId type = VectorOf(FloatType(), columns);
        result.type = type;
        result.id = Op(spv::OpMatrixTimesVector, type, {b.id, v.id});
        if (m_precise)
        {
            Decorate(result.id, spv::DecNoContraction);
        }
        return result;
    }
    if (Same(text, "sincos"))
    {
        if (!argCount(3) || !rvalues())
        {
            // The out arguments were loaded by rvalues(); reparse is not possible,
            // so sincos needs its lvalues untouched.
            return Val{};
        }
        (void)Fail(at, "sincos is not supported; use sin and cos");
        return Val{};
    }
    *handled = false;
    return Val{};
}

// ---- Methods --------------------------------------------------------------------

Val Compiler::CallMethod(const Val& object, const Name& method, Val* args, u32 count,
                         const Token& at) noexcept
{
    if (m_error)
    {
        return Val{};
    }
    if (object.kind != VK::Resource)
    {
        (void)Fail(at, "only resources have methods");
        return Val{};
    }
    const u32 declIndex = object.resource - 1;
    const Res kind = m_resources[declIndex].kind;
    char text[32] = {};
    if (method.length >= sizeof(text))
    {
        (void)Fail(at, "unknown method");
        return Val{};
    }
    std::memcpy(text, method.text, method.length);
    Val result{};
    result.kind = VK::R;

    // ---- Buffers ----
    if (kind == Res::StructuredBuffer || kind == Res::RWStructuredBuffer)
    {
        if (Same(text, "Load"))
        {
            if (count != 1)
            {
                (void)Fail(at, "Load takes an index");
                return Val{};
            }
            return Rvalue(Index(object, args[0], at));
        }
        if (Same(text, "GetDimensions"))
        {
            if (count != 2)
            {
                (void)Fail(at, "GetDimensions(out count, out stride)");
                return Val{};
            }
            const Val block = ResourceElement(object, at);
            if (m_error)
            {
                return Val{};
            }
            Val length{};
            length.kind = VK::R;
            length.type = UintType();
            length.id = Op(spv::OpArrayLength, UintType(), {block.id, 0});
            u32 size = 0;
            u32 align = 0;
            (void)SizeAndAlign(m_resources[declIndex].element, LayoutStructured, &size, &align);
            if (!Store(args[0], length) || !Store(args[1], ConstVal(UintType(), size)))
            {
                return Val{};
            }
            return Val{};
        }
        (void)Fail(at, "unknown structured buffer method");
        return Val{};
    }
    if (kind == Res::ByteAddressBuffer || kind == Res::RWByteAddressBuffer)
    {
        const Val block = ResourceElement(object, at);
        if (m_error)
        {
            return Val{};
        }
        const auto wordPointer = [&](const Val& address, u32 extra) noexcept -> Val
        {
            Val byteAddress = Convert(address, UintType());
            u32 word = Op(spv::OpShiftRightLogical, UintType(), {byteAddress.id, ConstUint(2)});
            if (extra != 0)
            {
                word = Op(spv::OpIAdd, UintType(), {word, ConstUint(extra)});
            }
            Val pointer{};
            pointer.kind = VK::Ptr;
            pointer.type = UintType();
            pointer.storage = spv::StorageBuffer;
            pointer.readOnly = kind == Res::ByteAddressBuffer;
            pointer.id = Op(spv::OpAccessChain, PointerTo(spv::StorageBuffer, UintType()),
                            {block.id, ConstUint(0), word});
            return pointer;
        };
        u32 width = 0;
        if (Same(text, "Load"))
            width = 1;
        else if (Same(text, "Load2"))
            width = 2;
        else if (Same(text, "Load3"))
            width = 3;
        else if (Same(text, "Load4"))
            width = 4;
        if (width != 0)
        {
            if (count != 1)
            {
                (void)Fail(at, "Load takes a byte address");
                return Val{};
            }
            u32 parts[4] = {};
            for (u32 component = 0; component < width; ++component)
            {
                parts[component] = Rvalue(wordPointer(args[0], component)).id;
            }
            result.type = VectorOf(UintType(), width);
            result.id =
                width == 1 ? parts[0] : OpN(spv::OpCompositeConstruct, result.type, parts, width);
            return result;
        }
        if (Same(text, "Store"))
            width = 1;
        else if (Same(text, "Store2"))
            width = 2;
        else if (Same(text, "Store3"))
            width = 3;
        else if (Same(text, "Store4"))
            width = 4;
        if (width != 0)
        {
            if (kind != Res::RWByteAddressBuffer || count != 2)
            {
                (void)Fail(at, "Store needs a RWByteAddressBuffer, an address and a value");
                return Val{};
            }
            const Val value = Convert(args[1], VectorOf(UintType(), width));
            for (u32 component = 0; component < width && !m_error; ++component)
            {
                Val part = value;
                if (width > 1)
                {
                    part.id = Op(spv::OpCompositeExtract, UintType(), {value.id, component});
                    part.type = UintType();
                }
                (void)Store(wordPointer(args[0], component), part);
            }
            return Val{};
        }
        if (std::strncmp(text, "Interlocked", 11) == 0)
        {
            if (kind != Res::RWByteAddressBuffer || count < 2)
            {
                (void)Fail(at, "interlocked methods need a RWByteAddressBuffer and an address");
                return Val{};
            }
            const Val pointer = wordPointer(args[0], 0);
            return Atomic(text, pointer, args + 1, count - 1, at);
        }
        if (Same(text, "GetDimensions"))
        {
            if (count != 1)
            {
                (void)Fail(at, "GetDimensions(out bytes)");
                return Val{};
            }
            const u32 words = Op(spv::OpArrayLength, UintType(), {block.id, 0});
            Val bytes{};
            bytes.kind = VK::R;
            bytes.type = UintType();
            bytes.id = Op(spv::OpIMul, UintType(), {words, ConstUint(4)});
            (void)Store(args[0], bytes);
            return Val{};
        }
        (void)Fail(at, "unknown byte address buffer method");
        return Val{};
    }

    // ---- Textures ----
    const ImageShape shape = ShapeOf(kind);
    if (!shape.valid)
    {
        (void)Fail(at, "this resource has no methods");
        return Val{};
    }
    const TypeId element = m_resources[declIndex].element;
    const TypeId scalar = ScalarOf(element);
    const TypeId texel = VectorOf(scalar, 4);
    const Val image = Load(ResourceElement(object, at));
    if (m_error)
    {
        return Val{};
    }
    const TypeId imageType = m_resources[declIndex].blockType;
    const auto finish = [&](u32 id) noexcept -> Val
    {
        Val raw{};
        raw.kind = VK::R;
        raw.type = texel;
        raw.id = id;
        return Convert(raw, element);
    };
    const auto sampled = [&](const Val& sampler) noexcept -> u32
    {
        if (sampler.kind != VK::Resource)
        {
            (void)Fail(at, "expected a sampler");
            return 0;
        }
        const Res samplerKind = m_resources[sampler.resource - 1].kind;
        if (samplerKind != Res::SamplerState && samplerKind != Res::SamplerComparisonState)
        {
            (void)Fail(at, "expected a sampler");
            return 0;
        }
        const Val handle = Load(ResourceElement(sampler, at));
        if (m_error)
        {
            return 0;
        }
        const u32 id =
            Op(spv::OpSampledImage, Intern(TK::SampledImage, imageType), {image.id, handle.id});
        if (object.nonUniform || sampler.nonUniform)
        {
            Decorate(id, spv::DecNonUniform);
        }
        return id;
    };
    const TypeId floatCoord = VectorOf(FloatType(), shape.coordComponents);

    if (Same(text, "Load"))
    {
        if (shape.storage)
        {
            if (count != 1)
            {
                (void)Fail(at, "Load takes a coordinate");
                return Val{};
            }
            return Rvalue(Index(object, args[0], at));
        }
        if (shape.dim == 3)
        {
            (void)Fail(at, "cube textures cannot be loaded");
            return Val{};
        }
        if (shape.ms)
        {
            if (count != 2)
            {
                (void)Fail(at, "Load(coord, sampleIndex)");
                return Val{};
            }
            const Val coord = Convert(args[0], VectorOf(IntType(), shape.coordComponents));
            const Val sample = Convert(args[1], IntType());
            return finish(Op(spv::OpImageFetch, texel, {image.id, coord.id, 0x40, sample.id}));
        }
        if (count != 1)
        {
            (void)Fail(at, "Load with an offset is not supported by the Vulkan HLSL compiler");
            return Val{};
        }
        const u32 n = shape.coordComponents;
        const Val full = Convert(args[0], VectorOf(IntType(), n + 1));
        if (m_error)
        {
            return Val{};
        }
        u32 coord = 0;
        if (n == 1)
        {
            coord = Op(spv::OpCompositeExtract, IntType(), {full.id, 0});
        }
        else
        {
            u32 operands[6] = {full.id, full.id, 0, 1, 2, 3};
            coord = OpN(spv::OpVectorShuffle, VectorOf(IntType(), n), operands, 2 + n);
        }
        const u32 lod = Op(spv::OpCompositeExtract, IntType(), {full.id, n});
        return finish(Op(spv::OpImageFetch, texel, {image.id, coord, 0x2, lod}));
    }
    if (Same(text, "Sample") || Same(text, "SampleBias") || Same(text, "SampleLevel") ||
        Same(text, "SampleGrad"))
    {
        if (shape.storage || shape.ms)
        {
            (void)Fail(at, "this texture cannot be sampled");
            return Val{};
        }
        const bool implicit = Same(text, "Sample") || Same(text, "SampleBias");
        const u32 wanted = Same(text, "Sample") ? 2U : Same(text, "SampleGrad") ? 4U : 3U;
        if (count != wanted)
        {
            (void)Fail(at, "wrong number of arguments (sampling with an offset or clamp is not "
                           "supported by the Vulkan HLSL compiler)");
            return Val{};
        }
        if (implicit && m_desc.stage != ShaderStage::Pixel)
        {
            (void)Fail(at, "Sample needs derivatives; use SampleLevel outside pixel shaders");
            return Val{};
        }
        const u32 combined = sampled(args[0]);
        const Val coord = Convert(args[1], floatCoord);
        if (m_error)
        {
            return Val{};
        }
        if (Same(text, "Sample"))
        {
            return finish(Op(spv::OpImageSampleImplicitLod, texel, {combined, coord.id}));
        }
        if (Same(text, "SampleBias"))
        {
            const Val bias = Convert(args[2], FloatType());
            return finish(
                Op(spv::OpImageSampleImplicitLod, texel, {combined, coord.id, 0x1, bias.id}));
        }
        if (Same(text, "SampleLevel"))
        {
            const Val lod = Convert(args[2], FloatType());
            return finish(
                Op(spv::OpImageSampleExplicitLod, texel, {combined, coord.id, 0x2, lod.id}));
        }
        const u32 gradComponents = shape.dim == 2 || shape.dim == 3 ? 3U : shape.dim == 0 ? 1U : 2U;
        const Val dx = Convert(args[2], VectorOf(FloatType(), gradComponents));
        const Val dy = Convert(args[3], VectorOf(FloatType(), gradComponents));
        return finish(
            Op(spv::OpImageSampleExplicitLod, texel, {combined, coord.id, 0x4, dx.id, dy.id}));
    }
    if (Same(text, "SampleCmp") || Same(text, "SampleCmpLevelZero"))
    {
        if (count != 3 || shape.storage || shape.ms)
        {
            (void)Fail(at, "SampleCmp(sampler, coord, compare)");
            return Val{};
        }
        const bool implicit = Same(text, "SampleCmp");
        if (implicit && m_desc.stage != ShaderStage::Pixel)
        {
            (void)Fail(at, "SampleCmp needs derivatives; use SampleCmpLevelZero outside pixel "
                           "shaders");
            return Val{};
        }
        const u32 combined = sampled(args[0]);
        const Val coord = Convert(args[1], floatCoord);
        const Val compare = Convert(args[2], FloatType());
        if (m_error)
        {
            return Val{};
        }
        result.type = FloatType();
        if (implicit)
        {
            result.id = Op(spv::OpImageSampleDrefImplicitLod, FloatType(),
                           {combined, coord.id, compare.id});
        }
        else
        {
            result.id = Op(spv::OpImageSampleDrefExplicitLod, FloatType(),
                           {combined, coord.id, compare.id, 0x2, ConstFloat(0.0F)});
        }
        return result;
    }
    if (Same(text, "Gather") || Same(text, "GatherRed") || Same(text, "GatherGreen") ||
        Same(text, "GatherBlue") || Same(text, "GatherAlpha"))
    {
        if (count != 2 || shape.storage || shape.ms || shape.dim == 0 || shape.dim == 2)
        {
            (void)Fail(at, "Gather(sampler, coord) on a 2D or cube texture");
            return Val{};
        }
        const u32 component = Same(text, "GatherGreen")   ? 1U
                              : Same(text, "GatherBlue")  ? 2U
                              : Same(text, "GatherAlpha") ? 3U
                                                          : 0U;
        const u32 combined = sampled(args[0]);
        const Val coord = Convert(args[1], floatCoord);
        if (m_error)
        {
            return Val{};
        }
        result.type = texel;
        result.id = Op(spv::OpImageGather, texel, {combined, coord.id, ConstUint(component)});
        return result;
    }
    if (Same(text, "GetDimensions"))
    {
        Capability(spv::CapImageQuery);
        const u32 n = shape.sizeComponents;
        const bool withMip = !shape.storage && !shape.ms && count == n + 2;
        if (!withMip && count != n)
        {
            (void)Fail(at, "unsupported GetDimensions overload");
            return Val{};
        }
        const TypeId sizeType = VectorOf(IntType(), n);
        u32 size = 0;
        u32 first = 0;
        if (shape.storage || shape.ms)
        {
            size = Op(spv::OpImageQuerySize, sizeType, {image.id});
        }
        else
        {
            u32 lod = ConstInt(0);
            if (withMip)
            {
                lod = Convert(args[0], IntType()).id;
                first = 1;
            }
            size = Op(spv::OpImageQuerySizeLod, sizeType, {image.id, lod});
        }
        for (u32 component = 0; component < n && !m_error; ++component)
        {
            Val part{};
            part.kind = VK::R;
            part.type = IntType();
            part.id = n == 1 ? size : Op(spv::OpCompositeExtract, IntType(), {size, component});
            (void)Store(args[first + component], part);
        }
        if (withMip && !m_error)
        {
            Val levels{};
            levels.kind = VK::R;
            levels.type = IntType();
            levels.id = Op(spv::OpImageQueryLevels, IntType(), {image.id});
            (void)Store(args[count - 1], levels);
        }
        return Val{};
    }
    (void)Fail(at, "unknown or unsupported texture method");
    return Val{};
}

}  // namespace cyber::rhi::detail::hlsl
