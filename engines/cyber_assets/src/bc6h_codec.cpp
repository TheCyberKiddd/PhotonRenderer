// BC6H (HDR) block codec (Phase 33). The decoder follows the Direct3D 11
// functional specification and is bit-exact with it; the tests compare it
// against the hardware decoder of every local adapter. The encoder searches
// the one-subset modes (11 to 14) and the two-subset modes (1 to 10) over the
// partitions, in the integer domain the decoder interpolates in, which is the
// half-float bit pattern scaled by 64/31, so squared error there is a relative
// (roughly logarithmic) error in the output.

#include <cyber/assets/bc_codec.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace cyber::assets
{

using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u16;
using cyber::core::u32;

namespace
{

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

// The 32 two-subset partitions of BC6H are the first 32 of BC7, as masks over
// the 16 texels (bit i set: texel i is in subset 1), and the anchor texel of
// subset 1 for each.
constexpr u16 kPartitions[32] = {
    0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80, 0xC800, 0xFFEC, 0xFE80,
    0xE800, 0xFFE8, 0xFF00, 0xFFF0, 0xF000, 0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310,
    0x3100, 0x8CCE, 0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C,
};
constexpr u32 kAnchors[32] = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
                              15, 2,  8,  2,  2,  8,  8,  15, 2,  8,  2,  2,  8,  8,  2,  2};

constexpr u32 kWeights3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
constexpr u32 kWeights4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

// The fields an endpoint is stored in. R, G and B each have a base (W), a
// second endpoint of subset 0 (X) and the two endpoints of subset 1 (Y, Z), in
// that order, so a field is `kRW + 4 * channel + k`.
enum Field : u32
{
    kM = 0,
    kD,
    kRW,
    kRX,
    kRY,
    kRZ,
    kGW,
    kGX,
    kGY,
    kGZ,
    kBW,
    kBX,
    kBY,
    kBZ,
    kFieldCount
};

constexpr u32 EndpointField(u32 channel, u32 k) noexcept
{
    return kRW + 4u * channel + k;
}

// `count` bits of `field` starting at bit `lo`, stored low bit first, or high
// bit first when `reversed` (the top bits of a wide base in modes 13 and 14).
struct Segment
{
    u32 field;
    u32 lo;
    u32 count;
    bool reversed;
};

constexpr Segment S(u32 field, u32 lo, u32 count) noexcept
{
    return Segment{field, lo, count, false};
}
constexpr Segment R(u32 field, u32 lo, u32 count) noexcept
{
    return Segment{field, lo, count, true};
}

struct Mode
{
    u32 value;     // the mode bits, low bit first
    u32 modeBits;  // 2 or 5
    u32 subsets;   // 1 or 2
    bool transformed;
    u32 epBits;  // bits of a base endpoint component
    u32 deltaBits[3];
    u32 segmentCount;
    Segment segments[30];
};

// Storage order of every mode, after the mode bits (Direct3D 11 functional
// specification, "BC6H Format").
constexpr Mode kModes[14] = {
    // 1: 10-bit base, 5 5 5 deltas.
    {0x00, 2, 2, true, 10, {5, 5, 5}, 21, {S(kGY, 4, 1),  S(kBY, 4, 1),  S(kBZ, 4, 1),
                                           S(kRW, 0, 10), S(kGW, 0, 10), S(kBW, 0, 10),
                                           S(kRX, 0, 5),  S(kGZ, 4, 1),  S(kGY, 0, 4),
                                           S(kGX, 0, 5),  S(kBZ, 0, 1),  S(kGZ, 0, 4),
                                           S(kBX, 0, 5),  S(kBZ, 1, 1),  S(kBY, 0, 4),
                                           S(kRY, 0, 5),  S(kBZ, 2, 1),  S(kRZ, 0, 5),
                                           S(kBZ, 3, 1),  S(kD, 0, 5)}},
    // 2: 7-bit base, 6 6 6.
    {0x01, 2, 2, true, 7, {6, 6, 6}, 25, {S(kGY, 5, 1), S(kGZ, 4, 1), S(kGZ, 5, 1), S(kRW, 0, 7),
                                          S(kBZ, 0, 1), S(kBZ, 1, 1), S(kBY, 4, 1), S(kGW, 0, 7),
                                          S(kBY, 5, 1), S(kBZ, 2, 1), S(kGY, 4, 1), S(kBW, 0, 7),
                                          S(kBZ, 3, 1), S(kBZ, 5, 1), S(kBZ, 4, 1), S(kRX, 0, 6),
                                          S(kGY, 0, 4), S(kGX, 0, 6), S(kGZ, 0, 4), S(kBX, 0, 6),
                                          S(kBY, 0, 4), S(kRY, 0, 6), S(kRZ, 0, 6), S(kD, 0, 5)}},
    // 3: 11-bit base, 5 4 4.
    {0x02,
     5,
     2,
     true,
     11,
     {5, 4, 4},
     21,
     {S(kRW, 0, 10), S(kGW, 0, 10), S(kBW, 0, 10), S(kRX, 0, 5), S(kRW, 10, 1), S(kGY, 0, 4),
      S(kGX, 0, 4), S(kGW, 10, 1), S(kBZ, 0, 1), S(kGZ, 0, 4), S(kBX, 0, 4), S(kBW, 10, 1),
      S(kBZ, 1, 1), S(kBY, 0, 4), S(kRY, 0, 5), S(kBZ, 2, 1), S(kRZ, 0, 5), S(kBZ, 3, 1),
      S(kD, 0, 5)}},
    // 4: 11-bit base, 4 5 4.
    {0x06, 5, 2, true, 11, {4, 5, 4}, 21, {S(kRW, 0, 10), S(kGW, 0, 10), S(kBW, 0, 10),
                                           S(kRX, 0, 4),  S(kRW, 10, 1), S(kGZ, 4, 1),
                                           S(kGY, 0, 4),  S(kGX, 0, 5),  S(kGW, 10, 1),
                                           S(kGZ, 0, 4),  S(kBX, 0, 4),  S(kBW, 10, 1),
                                           S(kBZ, 1, 1),  S(kBY, 0, 4),  S(kRY, 0, 4),
                                           S(kBZ, 0, 1),  S(kBZ, 2, 1),  S(kRZ, 0, 4),
                                           S(kGY, 4, 1),  S(kBZ, 3, 1),  S(kD, 0, 5)}},
    // 5: 11-bit base, 4 4 5.
    {0x0A, 5, 2, true, 11, {4, 4, 5}, 21, {S(kRW, 0, 10), S(kGW, 0, 10), S(kBW, 0, 10),
                                           S(kRX, 0, 4),  S(kRW, 10, 1), S(kBY, 4, 1),
                                           S(kGY, 0, 4),  S(kGX, 0, 4),  S(kGW, 10, 1),
                                           S(kBZ, 0, 1),  S(kGZ, 0, 4),  S(kBX, 0, 5),
                                           S(kBW, 10, 1), S(kBY, 0, 4),  S(kRY, 0, 4),
                                           S(kBZ, 1, 1),  S(kBZ, 2, 1),  S(kRZ, 0, 4),
                                           S(kBZ, 4, 1),  S(kBZ, 3, 1),  S(kD, 0, 5)}},
    // 6: 9-bit base, 5 5 5.
    {0x0E, 5, 2, true, 9, {5, 5, 5}, 21, {S(kRW, 0, 9), S(kBY, 4, 1), S(kGW, 0, 9), S(kGY, 4, 1),
                                          S(kBW, 0, 9), S(kBZ, 4, 1), S(kRX, 0, 5), S(kGZ, 4, 1),
                                          S(kGY, 0, 4), S(kGX, 0, 5), S(kBZ, 0, 1), S(kGZ, 0, 4),
                                          S(kBX, 0, 5), S(kBZ, 1, 1), S(kBY, 0, 4), S(kRY, 0, 5),
                                          S(kBZ, 2, 1), S(kRZ, 0, 5), S(kBZ, 3, 1), S(kD, 0, 5)}},
    // 7: 8-bit base, 6 5 5.
    {0x12, 5, 2, true, 8, {6, 5, 5}, 21, {S(kRW, 0, 8), S(kGZ, 4, 1), S(kBY, 4, 1), S(kGW, 0, 8),
                                          S(kBZ, 2, 1), S(kGY, 4, 1), S(kBW, 0, 8), S(kBZ, 3, 1),
                                          S(kBZ, 4, 1), S(kRX, 0, 6), S(kGY, 0, 4), S(kGX, 0, 5),
                                          S(kBZ, 0, 1), S(kGZ, 0, 4), S(kBX, 0, 5), S(kBZ, 1, 1),
                                          S(kBY, 0, 4), S(kRY, 0, 6), S(kRZ, 0, 6), S(kD, 0, 5)}},
    // 8: 8-bit base, 5 6 5.
    {0x16, 5, 2, true, 8, {5, 6, 5}, 24, {S(kRW, 0, 8), S(kBZ, 0, 1), S(kBY, 4, 1), S(kGW, 0, 8),
                                          S(kGY, 5, 1), S(kGY, 4, 1), S(kBW, 0, 8), S(kGZ, 5, 1),
                                          S(kBZ, 4, 1), S(kRX, 0, 5), S(kGZ, 4, 1), S(kGY, 0, 4),
                                          S(kGX, 0, 6), S(kGZ, 0, 4), S(kBX, 0, 5), S(kBZ, 1, 1),
                                          S(kBY, 0, 4), S(kRY, 0, 5), S(kBZ, 2, 1), S(kRZ, 0, 5),
                                          S(kBZ, 3, 1), S(kD, 0, 5)}},
    // 9: 8-bit base, 5 5 6.
    {0x1A, 5, 2, true, 8, {5, 5, 6}, 24, {S(kRW, 0, 8), S(kBZ, 1, 1), S(kBY, 4, 1), S(kGW, 0, 8),
                                          S(kBY, 5, 1), S(kGY, 4, 1), S(kBW, 0, 8), S(kBZ, 5, 1),
                                          S(kBZ, 4, 1), S(kRX, 0, 5), S(kGZ, 4, 1), S(kGY, 0, 4),
                                          S(kGX, 0, 5), S(kBZ, 0, 1), S(kGZ, 0, 4), S(kBX, 0, 6),
                                          S(kBY, 0, 4), S(kRY, 0, 5), S(kBZ, 2, 1), S(kRZ, 0, 5),
                                          S(kBZ, 3, 1), S(kD, 0, 5)}},
    // 10: four 6-bit endpoints, not transformed.
    {0x1E, 5, 2, false, 6, {6, 6, 6}, 26, {S(kRW, 0, 6), S(kGZ, 4, 1), S(kBZ, 0, 1), S(kBZ, 1, 1),
                                           S(kBY, 4, 1), S(kGW, 0, 6), S(kGY, 5, 1), S(kBY, 5, 1),
                                           S(kBZ, 2, 1), S(kGY, 4, 1), S(kBW, 0, 6), S(kGZ, 5, 1),
                                           S(kBZ, 3, 1), S(kBZ, 5, 1), S(kBZ, 4, 1), S(kRX, 0, 6),
                                           S(kGY, 0, 4), S(kGX, 0, 6), S(kGZ, 0, 4), S(kBX, 0, 6),
                                           S(kBY, 0, 4), S(kRY, 0, 6), S(kRZ, 0, 6), S(kD, 0, 5)}},
    // 11: one subset, two 10-bit endpoints, not transformed.
    {0x03,
     5,
     1,
     false,
     10,
     {10, 10, 10},
     6,
     {S(kRW, 0, 10), S(kGW, 0, 10), S(kBW, 0, 10), S(kRX, 0, 10), S(kGX, 0, 10), S(kBX, 0, 10)}},
    // 12: 11-bit base, 9-bit deltas.
    {0x07,
     5,
     1,
     true,
     11,
     {9, 9, 9},
     9,
     {S(kRW, 0, 10), S(kGW, 0, 10), S(kBW, 0, 10), S(kRX, 0, 9), S(kRW, 10, 1), S(kGX, 0, 9),
      S(kGW, 10, 1), S(kBX, 0, 9), S(kBW, 10, 1)}},
    // 13: 12-bit base, 8-bit deltas; the base's top bits are stored high bit first.
    {0x0B,
     5,
     1,
     true,
     12,
     {8, 8, 8},
     9,
     {S(kRW, 0, 10), S(kGW, 0, 10), S(kBW, 0, 10), S(kRX, 0, 8), R(kRW, 10, 2), S(kGX, 0, 8),
      R(kGW, 10, 2), S(kBX, 0, 8), R(kBW, 10, 2)}},
    // 14: 16-bit base, 4-bit deltas.
    {0x0F,
     5,
     1,
     true,
     16,
     {4, 4, 4},
     9,
     {S(kRW, 0, 10), S(kGW, 0, 10), S(kBW, 0, 10), S(kRX, 0, 4), R(kRW, 10, 6), S(kGX, 0, 4),
      R(kGW, 10, 6), S(kBX, 0, 4), R(kBW, 10, 6)}},
};

constexpr u32 CountHeaderBits(const Mode& mode) noexcept
{
    u32 bits = mode.modeBits;
    for (u32 i = 0; i < mode.segmentCount; ++i)
    {
        bits += mode.segments[i].count;
    }
    return bits;
}

// Every mode's header plus its indices fills the block exactly: 82 + 46 bits
// with two subsets (3-bit indices, two anchors) and 65 + 63 with one (4-bit
// indices, one anchor).
constexpr bool ModeTableIsComplete() noexcept
{
    for (const Mode& mode : kModes)
    {
        const u32 header = CountHeaderBits(mode);
        if (mode.subsets == 2 ? header != 82u : header != 65u)
        {
            return false;
        }
    }
    return true;
}
static_assert(ModeTableIsComplete(), "BC6H mode layouts must account for every header bit");

// ---------------------------------------------------------------------------
// Bit IO
// ---------------------------------------------------------------------------

class Reader
{
public:
    explicit Reader(const Byte* block) noexcept
    {
        std::memcpy(m_bytes, block, 16);
    }
    u32 Read(u32 count) noexcept
    {
        u32 value = 0;
        for (u32 bit = 0; bit < count; ++bit, ++m_position)
        {
            const u32 byteValue = static_cast<u32>(m_bytes[m_position >> 3u]);
            value |= ((byteValue >> (m_position & 7u)) & 1u) << bit;
        }
        return value;
    }

private:
    Byte m_bytes[16] = {};
    u32 m_position = 0;
};

class Writer
{
public:
    void Write(u32 value, u32 count) noexcept
    {
        for (u32 bit = 0; bit < count; ++bit, ++m_position)
        {
            if ((value >> bit) & 1u)
            {
                m_bytes[m_position >> 3u] = static_cast<Byte>(
                    static_cast<u32>(m_bytes[m_position >> 3u]) | (1u << (m_position & 7u)));
            }
        }
    }
    void CopyTo(Byte* block) const noexcept
    {
        std::memcpy(block, m_bytes, 16);
    }

private:
    Byte m_bytes[16] = {};
    u32 m_position = 0;
};

// ---------------------------------------------------------------------------
// Decoding arithmetic
// ---------------------------------------------------------------------------

i32 SignExtend(u32 value, u32 bits) noexcept
{
    const u32 shift = 32u - bits;
    return static_cast<i32>(value << shift) >> shift;
}

// An endpoint component code to the 16-bit integer the interpolation uses.
i32 Unquantize(i32 comp, u32 bits, bool isSigned) noexcept
{
    if (isSigned)
    {
        if (bits >= 16)
        {
            return comp;
        }
        const bool negative = comp < 0;
        if (negative)
        {
            comp = -comp;
        }
        i32 value;
        if (comp == 0)
        {
            value = 0;
        }
        else if (comp >= (1 << (bits - 1)) - 1)
        {
            value = 0x7FFF;
        }
        else
        {
            value = ((comp << 15) + 0x4000) >> (bits - 1);
        }
        return negative ? -value : value;
    }
    if (bits >= 15)
    {
        return comp;
    }
    if (comp == 0)
    {
        return 0;
    }
    if (comp == (1 << bits) - 1)
    {
        return 0xFFFF;
    }
    return ((comp << 16) + 0x8000) >> bits;
}

// The interpolated integer to a half-float bit pattern.
u32 FinishUnquantize(i32 comp, bool isSigned) noexcept
{
    if (isSigned)
    {
        return comp < 0 ? (static_cast<u32>((-comp) * 31) >> 5) | 0x8000u
                        : static_cast<u32>(comp * 31) >> 5;
    }
    return static_cast<u32>(comp * 31) >> 6;
}

const u32* WeightsFor(u32 subsets) noexcept
{
    return subsets == 2 ? kWeights3 : kWeights4;
}

u32 SubsetOfTexel(u32 subsets, u32 partition, u32 texel) noexcept
{
    return subsets == 2 ? (kPartitions[partition] >> texel) & 1u : 0u;
}

bool IsAnchorTexel(u32 subsets, u32 partition, u32 texel) noexcept
{
    return texel == 0 || (subsets == 2 && texel == kAnchors[partition]);
}

// ---------------------------------------------------------------------------
// Half floats
// ---------------------------------------------------------------------------

// Positive finite half bit patterns top out at 0x7BFF (65504).
constexpr u32 kMaxHalfMagnitude = 0x7BFF;

}  // namespace

f32 HalfToFloat(u16 half) noexcept
{
    const u32 sign = (static_cast<u32>(half) & 0x8000u) << 16u;
    const u32 exponent = (half >> 10u) & 0x1Fu;
    const u32 mantissa = half & 0x3FFu;
    u32 bits;
    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            bits = sign;
        }
        else
        {
            // A subnormal half is a normal float.
            u32 shift = 0;
            u32 m = mantissa;
            while ((m & 0x400u) == 0)
            {
                m <<= 1u;
                ++shift;
            }
            bits = sign | ((127u - 15u - shift + 1u) << 23u) | ((m & 0x3FFu) << 13u);
        }
    }
    else if (exponent == 31)
    {
        bits = sign | 0x7F800000u | (mantissa << 13u);
    }
    else
    {
        bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }
    f32 value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

u16 FloatToHalf(f32 value) noexcept
{
    u32 bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const u32 sign = (bits >> 16u) & 0x8000u;
    const u32 magnitude = bits & 0x7FFFFFFFu;
    if (magnitude > 0x7F800000u)
    {
        return static_cast<u16>(sign | 0x7E00u);  // NaN
    }
    if (magnitude >= 0x477FF000u)
    {
        // At or above the largest half after rounding: infinity.
        return static_cast<u16>(sign | 0x7C00u);
    }
    if (magnitude < 0x33000001u)
    {
        return static_cast<u16>(sign);  // rounds to zero
    }
    if (magnitude < 0x38800000u)
    {
        // A subnormal half: shift the implicit-one mantissa into place.
        const u32 exponent = magnitude >> 23u;
        const u32 mantissa = (magnitude & 0x7FFFFFu) | 0x800000u;
        const u32 shift = 126u - exponent;  // 14 .. 24
        const u32 half = mantissa >> shift;
        const u32 remainder = mantissa & ((1u << shift) - 1u);
        const u32 halfway = 1u << (shift - 1u);
        u32 result = half;
        if (remainder > halfway || (remainder == halfway && (half & 1u) != 0))
        {
            ++result;
        }
        return static_cast<u16>(sign | result);
    }
    // A normal half, rounding to nearest even.
    const u32 rebased = magnitude - 0x38000000u;
    u32 result = rebased >> 13u;
    const u32 remainder = rebased & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u) != 0))
    {
        ++result;
    }
    return static_cast<u16>(sign | result);
}

// ===========================================================================
// Decoder
// ===========================================================================

void DecodeBc6hBlock(const Byte* block, u16* rgb, bool isSigned) noexcept
{
    Reader reader(block);
    u32 value = reader.Read(2);
    const Mode* mode = nullptr;
    if (value < 2)
    {
        mode = &kModes[value];
    }
    else
    {
        value |= reader.Read(3) << 2u;
        for (u32 index = 2; index < 14; ++index)
        {
            if (kModes[index].value == value)
            {
                mode = &kModes[index];
                break;
            }
        }
    }
    if (mode == nullptr)
    {
        // Reserved modes decode to zero.
        std::memset(rgb, 0, 48 * sizeof(u16));
        return;
    }

    u32 fields[kFieldCount] = {};
    for (u32 index = 0; index < mode->segmentCount; ++index)
    {
        const Segment& segment = mode->segments[index];
        for (u32 k = 0; k < segment.count; ++k)
        {
            const u32 bit = segment.reversed ? segment.lo + segment.count - 1u - k : segment.lo + k;
            fields[segment.field] |= reader.Read(1) << bit;
        }
    }

    // Endpoints as component codes: E0 = W, E1 = X, and with two subsets E2 = Y,
    // E3 = Z.
    const u32 endpoints = mode->subsets * 2u;
    const u32 epMask = (1u << mode->epBits) - 1u;
    i32 code[4][3] = {};
    for (u32 channel = 0; channel < 3; ++channel)
    {
        const i32 base = isSigned ? SignExtend(fields[EndpointField(channel, 0)], mode->epBits)
                                  : static_cast<i32>(fields[EndpointField(channel, 0)]);
        code[0][channel] = base;
        for (u32 k = 1; k < endpoints; ++k)
        {
            const u32 raw = fields[EndpointField(channel, k)];
            if (mode->transformed)
            {
                const i32 delta = SignExtend(raw, mode->deltaBits[channel]);
                const u32 sum = static_cast<u32>(base + delta) & epMask;
                code[k][channel] = isSigned ? SignExtend(sum, mode->epBits) : static_cast<i32>(sum);
            }
            else
            {
                code[k][channel] =
                    isSigned ? SignExtend(raw, mode->epBits) : static_cast<i32>(raw & epMask);
            }
        }
    }
    i32 palette[4][3];
    for (u32 k = 0; k < endpoints; ++k)
    {
        for (u32 channel = 0; channel < 3; ++channel)
        {
            palette[k][channel] = Unquantize(code[k][channel], mode->epBits, isSigned);
        }
    }

    const u32 partition = mode->subsets == 2 ? fields[kD] : 0u;
    const u32 indexBits = mode->subsets == 2 ? 3u : 4u;
    const u32* weights = WeightsFor(mode->subsets);
    for (u32 texel = 0; texel < 16; ++texel)
    {
        const u32 bits =
            IsAnchorTexel(mode->subsets, partition, texel) ? indexBits - 1u : indexBits;
        const u32 index = reader.Read(bits);
        const u32 subset = SubsetOfTexel(mode->subsets, partition, texel);
        const u32 weight = weights[index];
        for (u32 channel = 0; channel < 3; ++channel)
        {
            const i32 a = palette[subset * 2u][channel];
            const i32 b = palette[subset * 2u + 1u][channel];
            const i32 mixed =
                (a * static_cast<i32>(64u - weight) + b * static_cast<i32>(weight) + 32) >> 6;
            rgb[texel * 3u + channel] = static_cast<u16>(FinishUnquantize(mixed, isSigned));
        }
    }
}

// ===========================================================================
// Encoder
// ===========================================================================

namespace
{

// A texel component as the interpolation domain value: unsigned half bits
// scale by 64/31 into 0..65535, signed magnitudes by 32/31 into -32767..32767.
f32 ToDomain(u16 half, bool isSigned) noexcept
{
    const u32 magnitude = std::min<u32>(half & 0x7FFFu, kMaxHalfMagnitude);
    // Non-finite values (above 0x7BFF) clamp to the largest finite half.
    if (isSigned)
    {
        const f32 value = static_cast<f32>(magnitude) * (32.0f / 31.0f);
        return (half & 0x8000u) != 0 ? -value : value;
    }
    if ((half & 0x8000u) != 0)
    {
        return 0.0f;  // negative values cannot be stored unsigned
    }
    return std::min(static_cast<f32>(magnitude) * (64.0f / 31.0f), 65535.0f);
}

// One texel of the block in the domain, three channels.
struct Point
{
    f32 v[3];
};

struct Fit
{
    f32 a[3];
    f32 b[3];
    f32 error;
};

f32 DomainMax(bool isSigned) noexcept
{
    return isSigned ? 32767.0f : 65535.0f;
}

f32 DomainMin(bool isSigned) noexcept
{
    return isSigned ? -32767.0f : 0.0f;
}

f32 SquaredDistance(const f32* p, const f32* q) noexcept
{
    const f32 dx = p[0] - q[0];
    const f32 dy = p[1] - q[1];
    const f32 dz = p[2] - q[2];
    return dx * dx + dy * dy + dz * dz;
}

// Least-squares endpoints of one subset against the weights of the index width,
// from the principal axis of its points refined by alternating index assignment
// and a 2x2 solve per channel.
Fit FitSubset(const Point* points, u32 count, const u32* weights, u32 levels,
              bool isSigned) noexcept
{
    Fit fit{};
    f32 mean[3] = {0, 0, 0};
    for (u32 i = 0; i < count; ++i)
    {
        for (u32 c = 0; c < 3; ++c)
        {
            mean[c] += points[i].v[c];
        }
    }
    for (u32 c = 0; c < 3; ++c)
    {
        mean[c] /= static_cast<f32>(count);
    }
    f32 cov[3][3] = {};
    for (u32 i = 0; i < count; ++i)
    {
        f32 d[3];
        for (u32 c = 0; c < 3; ++c)
        {
            d[c] = points[i].v[c] - mean[c];
        }
        for (u32 r = 0; r < 3; ++r)
        {
            for (u32 c = 0; c < 3; ++c)
            {
                cov[r][c] += d[r] * d[c];
            }
        }
    }
    f32 axis[3] = {1.0f, 1.0f, 1.0f};
    for (u32 iteration = 0; iteration < 8; ++iteration)
    {
        f32 next[3];
        for (u32 r = 0; r < 3; ++r)
        {
            next[r] = cov[r][0] * axis[0] + cov[r][1] * axis[1] + cov[r][2] * axis[2];
        }
        const f32 length = std::sqrt(next[0] * next[0] + next[1] * next[1] + next[2] * next[2]);
        if (length < 1.0e-6f)
        {
            break;
        }
        for (u32 c = 0; c < 3; ++c)
        {
            axis[c] = next[c] / length;
        }
    }
    f32 lowest = std::numeric_limits<f32>::max();
    f32 highest = std::numeric_limits<f32>::lowest();
    for (u32 i = 0; i < count; ++i)
    {
        f32 t = 0.0f;
        for (u32 c = 0; c < 3; ++c)
        {
            t += (points[i].v[c] - mean[c]) * axis[c];
        }
        lowest = std::min(lowest, t);
        highest = std::max(highest, t);
    }
    const f32 low = DomainMin(isSigned);
    const f32 high = DomainMax(isSigned);
    for (u32 c = 0; c < 3; ++c)
    {
        fit.a[c] = std::clamp(mean[c] + axis[c] * lowest, low, high);
        fit.b[c] = std::clamp(mean[c] + axis[c] * highest, low, high);
    }

    u32 chosen[16] = {};
    const auto assign = [&]() noexcept
    {
        f32 total = 0.0f;
        for (u32 i = 0; i < count; ++i)
        {
            f32 best = std::numeric_limits<f32>::max();
            for (u32 level = 0; level < levels; ++level)
            {
                const f32 w = static_cast<f32>(weights[level]) * (1.0f / 64.0f);
                f32 q[3];
                for (u32 c = 0; c < 3; ++c)
                {
                    q[c] = fit.a[c] + (fit.b[c] - fit.a[c]) * w;
                }
                const f32 distance = SquaredDistance(points[i].v, q);
                if (distance < best)
                {
                    best = distance;
                    chosen[i] = level;
                }
            }
            total += best;
        }
        return total;
    };
    fit.error = assign();
    for (u32 round = 0; round < 6; ++round)
    {
        f32 aa = 0, ab = 0, bb = 0;
        f32 ap[3] = {0, 0, 0};
        f32 bp[3] = {0, 0, 0};
        for (u32 i = 0; i < count; ++i)
        {
            const f32 w = static_cast<f32>(weights[chosen[i]]) * (1.0f / 64.0f);
            const f32 alpha = 1.0f - w;
            aa += alpha * alpha;
            ab += alpha * w;
            bb += w * w;
            for (u32 c = 0; c < 3; ++c)
            {
                ap[c] += alpha * points[i].v[c];
                bp[c] += w * points[i].v[c];
            }
        }
        const f32 determinant = aa * bb - ab * ab;
        if (determinant < 1.0e-6f)
        {
            break;
        }
        Fit next = fit;
        for (u32 c = 0; c < 3; ++c)
        {
            next.a[c] = std::clamp((bb * ap[c] - ab * bp[c]) / determinant, low, high);
            next.b[c] = std::clamp((aa * bp[c] - ab * ap[c]) / determinant, low, high);
        }
        const Fit saved = fit;
        fit = next;
        const f32 error = assign();
        if (error >= saved.error)
        {
            fit = saved;
            break;
        }
        fit.error = error;
    }
    return fit;
}

// The component code nearest to `value`, as a signed or unsigned code of `bits`.
i32 QuantizeComponent(f32 value, u32 bits, bool isSigned) noexcept
{
    const i32 lowCode = isSigned ? -(1 << (bits - 1)) : 0;
    const i32 highCode = isSigned ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;
    const f32 scale = isSigned ? static_cast<f32>(1 << (bits - 1)) / 32768.0f
                               : static_cast<f32>(1u << bits) / 65536.0f;
    const i32 guess = static_cast<i32>(std::floor(value * scale));
    i32 best = std::clamp(guess, lowCode, highCode);
    f32 bestError = std::numeric_limits<f32>::max();
    for (i32 candidate = guess - 1; candidate <= guess + 2; ++candidate)
    {
        const i32 code = std::clamp(candidate, lowCode, highCode);
        const f32 error = std::fabs(static_cast<f32>(Unquantize(code, bits, isSigned)) - value);
        if (error < bestError)
        {
            bestError = error;
            best = code;
        }
    }
    return best;
}

struct Candidate
{
    const Mode* mode = nullptr;
    u32 partition = 0;
    i32 code[4][3] = {};  // the codes the decoder will derive
    u32 index[16] = {};
    f32 error = std::numeric_limits<f32>::max();
};

// Builds the palettes the decoder will build from `candidate.code`, picks the
// best index of every texel against them (anchors restricted to the lower half)
// and sets the candidate's error.
void AssignIndices(Candidate& candidate, const Point* points, bool isSigned) noexcept
{
    const Mode& mode = *candidate.mode;
    const u32 subsets = mode.subsets;
    const u32 levels = subsets == 2 ? 8u : 16u;
    const u32* weights = WeightsFor(subsets);
    i32 unquantized[4][3];
    for (u32 k = 0; k < subsets * 2u; ++k)
    {
        for (u32 c = 0; c < 3; ++c)
        {
            unquantized[k][c] = Unquantize(candidate.code[k][c], mode.epBits, isSigned);
        }
    }
    f32 total = 0.0f;
    for (u32 texel = 0; texel < 16; ++texel)
    {
        const u32 subset = SubsetOfTexel(subsets, candidate.partition, texel);
        const u32 limit = IsAnchorTexel(subsets, candidate.partition, texel) ? levels / 2u : levels;
        f32 best = std::numeric_limits<f32>::max();
        u32 chosen = 0;
        for (u32 level = 0; level < limit; ++level)
        {
            f32 distance = 0.0f;
            for (u32 c = 0; c < 3; ++c)
            {
                const i32 a = unquantized[subset * 2u][c];
                const i32 b = unquantized[subset * 2u + 1u][c];
                const i32 mixed = (a * static_cast<i32>(64u - weights[level]) +
                                   b * static_cast<i32>(weights[level]) + 32) >>
                                  6;
                const f32 difference = static_cast<f32>(mixed) - points[texel].v[c];
                distance += difference * difference;
            }
            if (distance < best)
            {
                best = distance;
                chosen = level;
            }
        }
        candidate.index[texel] = chosen;
        total += best;
    }
    candidate.error = total;
}

// True when every endpoint code of `candidate` can be stored in its mode: a
// code is in the endpoint width's range, and a transformed endpoint differs
// from the base by a delta that fits (the difference wraps at the width).
bool CodesAreStorable(const Candidate& candidate, bool isSigned) noexcept
{
    const Mode& mode = *candidate.mode;
    const i32 low = isSigned ? -(1 << (mode.epBits - 1)) : 0;
    const i32 high =
        isSigned ? (1 << (mode.epBits - 1)) - 1 : static_cast<i32>((1u << mode.epBits) - 1u);
    for (u32 k = 0; k < mode.subsets * 2u; ++k)
    {
        for (u32 c = 0; c < 3; ++c)
        {
            if (candidate.code[k][c] < low || candidate.code[k][c] > high)
            {
                return false;
            }
            if (mode.transformed && k > 0)
            {
                const i32 difference = candidate.code[k][c] - candidate.code[0][c];
                const i32 limitLow = -(1 << (mode.deltaBits[c] - 1));
                const i32 limitHigh = (1 << (mode.deltaBits[c] - 1)) - 1;
                const i32 span = 1 << mode.epBits;
                const bool direct = difference >= limitLow && difference <= limitHigh;
                const i32 wrapped = difference > 0 ? difference - span : difference + span;
                const bool viaWrap = wrapped >= limitLow && wrapped <= limitHigh;
                if (!direct && !viaWrap)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

// Coordinate descent on the endpoint codes: each code moves by one or two steps
// either way while the block error falls and the mode can still store the codes.
void PolishCodes(Candidate& candidate, const Point* points, bool isSigned, u32 passes) noexcept
{
    const u32 endpoints = candidate.mode->subsets * 2u;
    for (u32 pass = 0; pass < passes; ++pass)
    {
        bool improved = false;
        for (u32 k = 0; k < endpoints; ++k)
        {
            for (u32 c = 0; c < 3; ++c)
            {
                for (const i32 step : {-2, -1, 1, 2})
                {
                    Candidate trial = candidate;
                    trial.code[k][c] += step;
                    if (!CodesAreStorable(trial, isSigned))
                    {
                        continue;
                    }
                    AssignIndices(trial, points, isSigned);
                    if (trial.error < candidate.error)
                    {
                        candidate = trial;
                        improved = true;
                    }
                }
            }
        }
        if (!improved)
        {
            break;
        }
    }
}

// Evaluates `mode` with `partition` on the block, given the unquantized subset
// fits for that partition. Quantizes, applies the transform and its range, then
// re-assigns the indices against the exact palette the decoder will build.
bool TryMode(const Mode& mode, u32 partition, const Fit* fits, const Point* points, bool isSigned,
             Candidate& out) noexcept
{
    const u32 subsets = mode.subsets;
    const u32 indexBits = subsets == 2 ? 3u : 4u;
    const u32 levels = 1u << indexBits;
    const u32* weights = WeightsFor(subsets);
    const u32 anchorLimit = levels / 2u;  // an anchor index has a zero top bit

    // Orient each subset so its anchor texel lands in the lower half of the
    // weights, before quantizing: a flipped subset swaps its endpoints.
    f32 ends[4][3];
    for (u32 subset = 0; subset < subsets; ++subset)
    {
        f32 a[3];
        f32 b[3];
        for (u32 c = 0; c < 3; ++c)
        {
            a[c] = fits[subset].a[c];
            b[c] = fits[subset].b[c];
        }
        u32 anchorTexel = 0;
        if (subset == 1)
        {
            anchorTexel = kAnchors[partition];
        }
        f32 best = std::numeric_limits<f32>::max();
        u32 anchorIndex = 0;
        for (u32 level = 0; level < levels; ++level)
        {
            const f32 w = static_cast<f32>(weights[level]) * (1.0f / 64.0f);
            f32 q[3];
            for (u32 c = 0; c < 3; ++c)
            {
                q[c] = a[c] + (b[c] - a[c]) * w;
            }
            const f32 distance = SquaredDistance(points[anchorTexel].v, q);
            if (distance < best)
            {
                best = distance;
                anchorIndex = level;
            }
        }
        if (anchorIndex >= anchorLimit)
        {
            for (u32 c = 0; c < 3; ++c)
            {
                std::swap(a[c], b[c]);
            }
        }
        for (u32 c = 0; c < 3; ++c)
        {
            ends[subset * 2u][c] = a[c];
            ends[subset * 2u + 1u][c] = b[c];
        }
    }

    // Quantize, then apply the delta transform and its limits.
    const u32 endpoints = subsets * 2u;
    const u32 epMask = (1u << mode.epBits) - 1u;
    for (u32 c = 0; c < 3; ++c)
    {
        i32 quantized[4];
        for (u32 k = 0; k < endpoints; ++k)
        {
            quantized[k] = QuantizeComponent(ends[k][c], mode.epBits, isSigned);
        }
        out.code[0][c] = quantized[0];
        for (u32 k = 1; k < endpoints; ++k)
        {
            if (!mode.transformed)
            {
                out.code[k][c] = quantized[k];
                continue;
            }
            const i32 limitLow = -(1 << (mode.deltaBits[c] - 1));
            const i32 limitHigh = (1 << (mode.deltaBits[c] - 1)) - 1;
            i32 delta = quantized[k] - quantized[0];
            if (delta < limitLow || delta > limitHigh)
            {
                // The sum wraps modulo the endpoint width, so a delta may also
                // reach the target the other way round.
                const i32 span = 1 << mode.epBits;
                const i32 wrapped = delta > 0 ? delta - span : delta + span;
                delta = (wrapped >= limitLow && wrapped <= limitHigh)
                            ? wrapped
                            : std::clamp(delta, limitLow, limitHigh);
            }
            const u32 sum = static_cast<u32>(quantized[0] + delta) & epMask;
            out.code[k][c] = isSigned ? SignExtend(sum, mode.epBits) : static_cast<i32>(sum);
        }
    }

    out.mode = &mode;
    out.partition = partition;
    AssignIndices(out, points, isSigned);
    return true;
}

void WriteBlock(const Candidate& candidate, Byte* block) noexcept
{
    const Mode& mode = *candidate.mode;
    const u32 endpoints = mode.subsets * 2u;
    const u32 epMask = (1u << mode.epBits) - 1u;
    u32 fields[kFieldCount] = {};
    fields[kD] = candidate.partition;
    for (u32 c = 0; c < 3; ++c)
    {
        fields[EndpointField(c, 0)] = static_cast<u32>(candidate.code[0][c]) & epMask;
        for (u32 k = 1; k < endpoints; ++k)
        {
            if (mode.transformed)
            {
                const u32 deltaMask = (1u << mode.deltaBits[c]) - 1u;
                const i32 delta = SignExtend(
                    static_cast<u32>(candidate.code[k][c] - candidate.code[0][c]) & epMask,
                    mode.epBits);
                fields[EndpointField(c, k)] = static_cast<u32>(delta) & deltaMask;
            }
            else
            {
                fields[EndpointField(c, k)] = static_cast<u32>(candidate.code[k][c]) & epMask;
            }
        }
    }
    Writer writer;
    writer.Write(mode.value, mode.modeBits);
    for (u32 index = 0; index < mode.segmentCount; ++index)
    {
        const Segment& segment = mode.segments[index];
        for (u32 k = 0; k < segment.count; ++k)
        {
            const u32 bit = segment.reversed ? segment.lo + segment.count - 1u - k : segment.lo + k;
            writer.Write((fields[segment.field] >> bit) & 1u, 1);
        }
    }
    const u32 indexBits = mode.subsets == 2 ? 3u : 4u;
    for (u32 texel = 0; texel < 16; ++texel)
    {
        const bool anchor = IsAnchorTexel(mode.subsets, candidate.partition, texel);
        writer.Write(candidate.index[texel], anchor ? indexBits - 1u : indexBits);
    }
    writer.CopyTo(block);
}

}  // namespace

void EncodeBc6hBlock(const u16* rgb, Byte* block, const BcEncodeSettings& settings,
                     bool isSigned) noexcept
{
    Point points[16];
    for (u32 texel = 0; texel < 16; ++texel)
    {
        for (u32 c = 0; c < 3; ++c)
        {
            points[texel].v[c] = ToDomain(rgb[texel * 3u + c], isSigned);
        }
    }

    // The best few candidates by block error, polished at the end: the ranking is
    // by the unpolished error, but a close runner-up can overtake after polishing.
    constexpr u32 kMaxPool = 4;
    Candidate pool[kMaxPool];
    u32 poolSize = 0;
    const u32 poolLimit = settings.quality == BcQuality::Fast     ? 1u
                          : settings.quality == BcQuality::Normal ? 2u
                                                                  : kMaxPool;
    const auto consider = [&](const Candidate& candidate) noexcept
    {
        if (candidate.mode == nullptr ||
            (poolSize == poolLimit && candidate.error >= pool[poolSize - 1u].error))
        {
            return;
        }
        u32 slot = poolSize < poolLimit ? poolSize++ : poolSize - 1u;
        while (slot > 0 && pool[slot - 1u].error > candidate.error)
        {
            pool[slot] = pool[slot - 1u];
            --slot;
        }
        pool[slot] = candidate;
    };
    const auto finish = [&]() noexcept
    {
        const u32 passes = settings.quality == BcQuality::Fast
                               ? 0u
                               : (settings.quality == BcQuality::Normal ? 2u : 4u);
        u32 bestSlot = 0;
        for (u32 slot = 0; slot < poolSize; ++slot)
        {
            if (passes > 0)
            {
                PolishCodes(pool[slot], points, isSigned, passes);
            }
            if (pool[slot].error < pool[bestSlot].error)
            {
                bestSlot = slot;
            }
        }
        WriteBlock(pool[bestSlot], block);
    };

    // One subset: modes 11 to 14 (table indices 10 to 13) share one fit with
    // 4-bit indices. Mode 11 takes any block; Fast stops after modes 11 and 12.
    {
        const Fit fit = FitSubset(points, 16, kWeights4, 16, isSigned);
        const u32 last = settings.quality == BcQuality::Fast ? 12u : 14u;
        for (u32 index = 10; index < last; ++index)
        {
            Candidate candidate;
            if (TryMode(kModes[index], 0, &fit, points, isSigned, candidate))
            {
                consider(candidate);
            }
        }
    }
    if (settings.quality == BcQuality::Fast)
    {
        finish();
        return;
    }

    // Two subsets: rank the 32 partitions by the unquantized fit of their
    // subsets, and try every two-subset mode on the best few.
    struct Ranked
    {
        f32 error;
        u32 partition;
        Fit fits[2];
    };
    Ranked ranked[32];
    for (u32 partition = 0; partition < 32; ++partition)
    {
        Point groups[2][16];
        u32 counts[2] = {0, 0};
        for (u32 texel = 0; texel < 16; ++texel)
        {
            const u32 subset = SubsetOfTexel(2, partition, texel);
            groups[subset][counts[subset]++] = points[texel];
        }
        Ranked& entry = ranked[partition];
        entry.partition = partition;
        entry.error = 0.0f;
        for (u32 subset = 0; subset < 2; ++subset)
        {
            entry.fits[subset] = FitSubset(groups[subset], counts[subset], kWeights3, 8, isSigned);
            entry.error += entry.fits[subset].error;
        }
    }
    std::sort(ranked, ranked + 32,
              [](const Ranked& lhs, const Ranked& rhs) noexcept {
                  return lhs.error < rhs.error ||
                         (lhs.error == rhs.error && lhs.partition < rhs.partition);
              });
    const u32 wanted = settings.quality == BcQuality::Normal ? 4u : 32u;
    for (u32 rank = 0; rank < wanted; ++rank)
    {
        for (u32 index = 0; index < 10; ++index)
        {
            Candidate candidate;
            if (TryMode(kModes[index], ranked[rank].partition, ranked[rank].fits, points, isSigned,
                        candidate))
            {
                consider(candidate);
            }
        }
    }
    finish();
}

bool CompressBc6hSurface(Span<const u16> rgbaHalf, u32 width, u32 height, Span<Byte> blocks,
                         const BcEncodeSettings& settings, bool isSigned) noexcept
{
    if (width == 0 || height == 0 || rgbaHalf.Size() != static_cast<usize>(width) * height * 4u ||
        blocks.Size() != Bc6hSurfaceBytes(width, height))
    {
        return false;
    }
    const u32 blocksX = (width + 3u) / 4u;
    const u32 blocksY = (height + 3u) / 4u;
    for (u32 by = 0; by < blocksY; ++by)
    {
        for (u32 bx = 0; bx < blocksX; ++bx)
        {
            u16 texels[kBc6hBlockValues];
            for (u32 y = 0; y < 4; ++y)
            {
                const u32 sy = std::min(by * 4u + y, height - 1u);
                for (u32 x = 0; x < 4; ++x)
                {
                    const u32 sx = std::min(bx * 4u + x, width - 1u);
                    const u16* texel = rgbaHalf.Data() + (static_cast<usize>(sy) * width + sx) * 4u;
                    for (u32 c = 0; c < 3; ++c)
                    {
                        texels[(y * 4u + x) * 3u + c] = texel[c];
                    }
                }
            }
            EncodeBc6hBlock(
                texels, blocks.Data() + (static_cast<usize>(by) * blocksX + bx) * kBc6hBlockBytes,
                settings, isSigned);
        }
    }
    return true;
}

bool DecompressBc6hSurface(Span<const Byte> blocks, u32 width, u32 height, Span<u16> rgbaHalf,
                           bool isSigned) noexcept
{
    if (width == 0 || height == 0 || rgbaHalf.Size() != static_cast<usize>(width) * height * 4u ||
        blocks.Size() != Bc6hSurfaceBytes(width, height))
    {
        return false;
    }
    const u32 blocksX = (width + 3u) / 4u;
    const u32 blocksY = (height + 3u) / 4u;
    for (u32 by = 0; by < blocksY; ++by)
    {
        for (u32 bx = 0; bx < blocksX; ++bx)
        {
            u16 texels[kBc6hBlockValues];
            DecodeBc6hBlock(blocks.Data() +
                                (static_cast<usize>(by) * blocksX + bx) * kBc6hBlockBytes,
                            texels, isSigned);
            for (u32 y = 0; y < 4 && by * 4u + y < height; ++y)
            {
                for (u32 x = 0; x < 4 && bx * 4u + x < width; ++x)
                {
                    u16* texel = rgbaHalf.Data() +
                                 (static_cast<usize>(by * 4u + y) * width + bx * 4u + x) * 4u;
                    for (u32 c = 0; c < 3; ++c)
                    {
                        texel[c] = texels[(y * 4u + x) * 3u + c];
                    }
                    texel[3] = 0x3C00;  // 1.0
                }
            }
        }
    }
    return true;
}

}  // namespace cyber::assets
