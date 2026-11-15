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
using cyber::core::u64;

namespace
{

// ============================================================================
// Shared helpers
// ============================================================================

u32 ReadLe32(const Byte* bytes) noexcept
{
    return static_cast<u32>(bytes[0]) | (static_cast<u32>(bytes[1]) << 8u) |
           (static_cast<u32>(bytes[2]) << 16u) | (static_cast<u32>(bytes[3]) << 24u);
}

void WriteLe16(Byte* bytes, u32 value) noexcept
{
    bytes[0] = static_cast<Byte>(value & 0xFFu);
    bytes[1] = static_cast<Byte>((value >> 8u) & 0xFFu);
}

void WriteLe32(Byte* bytes, u32 value) noexcept
{
    WriteLe16(bytes, value & 0xFFFFu);
    WriteLe16(bytes + 2, value >> 16u);
}

i32 Clamp255(i32 value) noexcept
{
    return value < 0 ? 0 : (value > 255 ? 255 : value);
}

f32 Clamp255f(f32 value) noexcept
{
    return value < 0.0f ? 0.0f : (value > 255.0f ? 255.0f : value);
}

u32 Square(i32 value) noexcept
{
    return static_cast<u32>(value * value);
}

// Principal axis of a point cloud by power iteration on its covariance. `dims`
// is 1..4. Returns false when the cloud has no extent (all points equal).
bool PrincipalAxis(const f32 covariance[4][4], u32 dims, f32 axis[4]) noexcept
{
    // Start from the channel with the largest variance, which converges in a
    // few steps for the elongated clouds texture blocks almost always are.
    u32 largest = 0;
    for (u32 d = 1; d < dims; ++d)
    {
        if (covariance[d][d] > covariance[largest][largest])
        {
            largest = d;
        }
    }
    if (!(covariance[largest][largest] > 1.0e-6f))
    {
        return false;
    }
    f32 vector[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 d = 0; d < dims; ++d)
    {
        vector[d] = covariance[largest][d];
    }
    for (u32 iteration = 0; iteration < 8; ++iteration)
    {
        f32 next[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        f32 length = 0.0f;
        for (u32 row = 0; row < dims; ++row)
        {
            for (u32 column = 0; column < dims; ++column)
            {
                next[row] += covariance[row][column] * vector[column];
            }
            length = std::max(length, std::fabs(next[row]));
        }
        if (!(length > 0.0f))
        {
            return false;
        }
        for (u32 d = 0; d < dims; ++d)
        {
            vector[d] = next[d] / length;
        }
    }
    f32 norm = 0.0f;
    for (u32 d = 0; d < dims; ++d)
    {
        norm += vector[d] * vector[d];
    }
    norm = std::sqrt(norm);
    if (!(norm > 0.0f))
    {
        return false;
    }
    for (u32 d = 0; d < 4; ++d)
    {
        axis[d] = d < dims ? vector[d] / norm : 0.0f;
    }
    return true;
}

// Fits a line through `count` points of `dims` channels and returns its two
// extreme projections as endpoint estimates. Degenerate clouds return the mean
// for both endpoints.
void LineFitEndpoints(const f32 (*points)[4], u32 count, u32 dims, f32 low[4], f32 high[4]) noexcept
{
    f32 mean[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 i = 0; i < count; ++i)
    {
        for (u32 d = 0; d < dims; ++d)
        {
            mean[d] += points[i][d];
        }
    }
    const f32 inverse = count > 0 ? 1.0f / static_cast<f32>(count) : 0.0f;
    for (u32 d = 0; d < dims; ++d)
    {
        mean[d] *= inverse;
    }
    f32 covariance[4][4] = {};
    for (u32 i = 0; i < count; ++i)
    {
        for (u32 r = 0; r < dims; ++r)
        {
            const f32 dr = points[i][r] - mean[r];
            for (u32 c = r; c < dims; ++c)
            {
                covariance[r][c] += dr * (points[i][c] - mean[c]);
            }
        }
    }
    for (u32 r = 0; r < dims; ++r)
    {
        for (u32 c = 0; c < r; ++c)
        {
            covariance[r][c] = covariance[c][r];
        }
    }
    f32 axis[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 d = 0; d < 4; ++d)
    {
        low[d] = d < dims ? mean[d] : 0.0f;
        high[d] = low[d];
    }
    if (!PrincipalAxis(covariance, dims, axis))
    {
        return;
    }
    f32 minimum = std::numeric_limits<f32>::max();
    f32 maximum = -std::numeric_limits<f32>::max();
    for (u32 i = 0; i < count; ++i)
    {
        f32 t = 0.0f;
        for (u32 d = 0; d < dims; ++d)
        {
            t += (points[i][d] - mean[d]) * axis[d];
        }
        minimum = std::min(minimum, t);
        maximum = std::max(maximum, t);
    }
    for (u32 d = 0; d < dims; ++d)
    {
        low[d] = Clamp255f(mean[d] + axis[d] * minimum);
        high[d] = Clamp255f(mean[d] + axis[d] * maximum);
    }
}

// Sum of squared distances from the points to their best-fit line: the total
// variance minus the variance along the principal axis. A cheap estimate of
// how well one endpoint pair can represent a subset, used to rank partitions.
f32 LineFitResidual(const f32 (*points)[4], u32 count, u32 dims) noexcept
{
    if (count < 2)
    {
        return 0.0f;
    }
    f32 mean[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 i = 0; i < count; ++i)
    {
        for (u32 d = 0; d < dims; ++d)
        {
            mean[d] += points[i][d];
        }
    }
    for (u32 d = 0; d < dims; ++d)
    {
        mean[d] /= static_cast<f32>(count);
    }
    f32 covariance[4][4] = {};
    f32 total = 0.0f;
    for (u32 i = 0; i < count; ++i)
    {
        for (u32 r = 0; r < dims; ++r)
        {
            const f32 dr = points[i][r] - mean[r];
            total += dr * dr;
            for (u32 c = 0; c < dims; ++c)
            {
                covariance[r][c] += dr * (points[i][c] - mean[c]);
            }
        }
    }
    f32 axis[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (!PrincipalAxis(covariance, dims, axis))
    {
        return total;
    }
    f32 along = 0.0f;
    for (u32 r = 0; r < dims; ++r)
    {
        f32 row = 0.0f;
        for (u32 c = 0; c < dims; ++c)
        {
            row += covariance[r][c] * axis[c];
        }
        along += axis[r] * row;
    }
    return std::max(total - along, 0.0f);
}

u32 RefinementPasses(BcQuality quality) noexcept
{
    switch (quality)
    {
    case BcQuality::Fast:
        return 1;
    case BcQuality::Normal:
        return 2;
    case BcQuality::High:
        return 4;
    }
    return 2;
}

// ============================================================================
// BC1
// ============================================================================

struct Rgb565
{
    i32 r = 0;
    i32 g = 0;
    i32 b = 0;
};

i32 Expand5(i32 value) noexcept
{
    return (value << 3) | (value >> 2);
}

i32 Expand6(i32 value) noexcept
{
    return (value << 2) | (value >> 4);
}

u32 Pack565(const Rgb565& c) noexcept
{
    return (static_cast<u32>(c.r) << 11u) | (static_cast<u32>(c.g) << 5u) | static_cast<u32>(c.b);
}

Rgb565 Unpack565(u32 packed) noexcept
{
    return Rgb565{static_cast<i32>((packed >> 11u) & 31u), static_cast<i32>((packed >> 5u) & 63u),
                  static_cast<i32>(packed & 31u)};
}

// The decoder's palette, shared by the encoder so it scores exactly what will
// be reconstructed.
void Bc1Palette(u32 c0, u32 c1, i32 palette[4][4]) noexcept
{
    const Rgb565 a = Unpack565(c0);
    const Rgb565 b = Unpack565(c1);
    const i32 e0[3] = {Expand5(a.r), Expand6(a.g), Expand5(a.b)};
    const i32 e1[3] = {Expand5(b.r), Expand6(b.g), Expand5(b.b)};
    for (u32 ch = 0; ch < 3; ++ch)
    {
        palette[0][ch] = e0[ch];
        palette[1][ch] = e1[ch];
        if (c0 > c1)
        {
            palette[2][ch] = (2 * e0[ch] + e1[ch] + 1) / 3;
            palette[3][ch] = (e0[ch] + 2 * e1[ch] + 1) / 3;
        }
        else
        {
            palette[2][ch] = (e0[ch] + e1[ch] + 1) / 2;
            palette[3][ch] = 0;
        }
    }
    palette[0][3] = 255;
    palette[1][3] = 255;
    palette[2][3] = 255;
    palette[3][3] = c0 > c1 ? 255 : 0;
}

// Optimal single-colour endpoint pairs: for every 8-bit value, the 5- or 6-bit
// endpoints whose two-thirds interpolant reproduces it most closely. A solid
// block encoded this way is usually exact, where a plain quantization to 565
// loses up to four codes.
struct SingleColorTables
{
    u8 match5[256][2] = {};
    u8 match6[256][2] = {};

    SingleColorTables() noexcept
    {
        Build(match5, 31, &Expand5);
        Build(match6, 63, &Expand6);
    }

    static void Build(u8 (&table)[256][2], i32 maximum, i32 (*expand)(i32)) noexcept
    {
        for (i32 value = 0; value < 256; ++value)
        {
            i32 bestError = 1 << 30;
            for (i32 a = 0; a <= maximum; ++a)
            {
                for (i32 b = 0; b <= maximum; ++b)
                {
                    const i32 interpolated = (2 * expand(a) + expand(b) + 1) / 3;
                    // Prefer pairs close together: they survive the endpoint
                    // ordering swap without changing the interpolant much.
                    const i32 error = std::abs(interpolated - value) * 100 + std::abs(a - b);
                    if (error < bestError)
                    {
                        bestError = error;
                        table[value][0] = static_cast<u8>(a);
                        table[value][1] = static_cast<u8>(b);
                    }
                }
            }
        }
    }
};

const SingleColorTables& GetSingleColorTables() noexcept
{
    static const SingleColorTables tables{};
    return tables;
}

Rgb565 Quantize565(const f32 color[3]) noexcept
{
    const auto quantize = [](f32 value, i32 maximum) noexcept
    {
        const i32 q =
            static_cast<i32>(Clamp255f(value) * static_cast<f32>(maximum) / 255.0f + 0.5f);
        return std::clamp(q, 0, maximum);
    };
    return Rgb565{quantize(color[0], 31), quantize(color[1], 63), quantize(color[2], 31)};
}

struct Bc1Candidate
{
    u32 c0 = 0;
    u32 c1 = 0;
    u32 indices = 0;
    u32 error = std::numeric_limits<u32>::max();
};

// Assigns every texel to its nearest palette entry and returns the RGB squared
// error. Transparent texels take index 3 at no cost; in three-colour mode
// opaque texels may not use index 3, which decodes transparent.
Bc1Candidate ScoreBc1(const u8* rgba, const bool* transparent, u32 c0, u32 c1) noexcept
{
    Bc1Candidate candidate;
    candidate.c0 = c0;
    candidate.c1 = c1;
    i32 palette[4][4];
    Bc1Palette(c0, c1, palette);
    const bool fourColor = c0 > c1;
    const u32 usable = fourColor ? 4u : 3u;
    u32 error = 0;
    u32 indices = 0;
    for (u32 texel = 0; texel < 16; ++texel)
    {
        u32 best = 0;
        if (transparent[texel])
        {
            best = 3;
        }
        else
        {
            u32 bestError = std::numeric_limits<u32>::max();
            for (u32 entry = 0; entry < usable; ++entry)
            {
                const u32 e = Square(palette[entry][0] - rgba[texel * 4 + 0]) +
                              Square(palette[entry][1] - rgba[texel * 4 + 1]) +
                              Square(palette[entry][2] - rgba[texel * 4 + 2]);
                if (e < bestError)
                {
                    bestError = e;
                    best = entry;
                }
            }
            error += bestError;
        }
        indices |= best << (texel * 2u);
    }
    candidate.indices = indices;
    candidate.error = error;
    return candidate;
}

// Orders the endpoints for the requested mode, then scores. Four-colour mode
// needs c0 > c1 and three-colour mode c0 <= c1; equal endpoints in four-colour
// mode fall back to three-colour, which still reproduces the colour exactly.
Bc1Candidate ScoreBc1Mode(const u8* rgba, const bool* transparent, Rgb565 a, Rgb565 b,
                          bool fourColor) noexcept
{
    u32 c0 = Pack565(a);
    u32 c1 = Pack565(b);
    if (fourColor ? c0 < c1 : c0 > c1)
    {
        std::swap(c0, c1);
    }
    return ScoreBc1(rgba, transparent, c0, c1);
}

// Least-squares endpoint refit for fixed indices. Each texel contributes
// x = w * e0 + (1 - w) * e1 with w taken from its palette slot.
bool RefitBc1(const u8* rgba, const bool* transparent, const Bc1Candidate& candidate, f32 e0[3],
              f32 e1[3]) noexcept
{
    const bool fourColor = candidate.c0 > candidate.c1;
    const f32 weights4[4] = {1.0f, 0.0f, 2.0f / 3.0f, 1.0f / 3.0f};
    const f32 weights3[4] = {1.0f, 0.0f, 0.5f, 0.0f};
    f32 aa = 0.0f;
    f32 ab = 0.0f;
    f32 bb = 0.0f;
    f32 ax[3] = {0.0f, 0.0f, 0.0f};
    f32 bx[3] = {0.0f, 0.0f, 0.0f};
    for (u32 texel = 0; texel < 16; ++texel)
    {
        if (transparent[texel])
        {
            continue;
        }
        const u32 index = (candidate.indices >> (texel * 2u)) & 3u;
        const f32 w = fourColor ? weights4[index] : weights3[index];
        const f32 v = 1.0f - w;
        aa += w * w;
        ab += w * v;
        bb += v * v;
        for (u32 ch = 0; ch < 3; ++ch)
        {
            const f32 x = static_cast<f32>(rgba[texel * 4 + ch]);
            ax[ch] += w * x;
            bx[ch] += v * x;
        }
    }
    const f32 determinant = aa * bb - ab * ab;
    if (std::fabs(determinant) < 1.0e-6f)
    {
        return false;
    }
    const f32 inverse = 1.0f / determinant;
    for (u32 ch = 0; ch < 3; ++ch)
    {
        e0[ch] = Clamp255f((bb * ax[ch] - ab * bx[ch]) * inverse);
        e1[ch] = Clamp255f((aa * bx[ch] - ab * ax[ch]) * inverse);
    }
    return true;
}

void WriteBc1(Byte* block, const Bc1Candidate& candidate) noexcept
{
    WriteLe16(block, candidate.c0);
    WriteLe16(block + 2, candidate.c1);
    WriteLe32(block + 4, candidate.indices);
}

// ============================================================================
// BC4
// ============================================================================

void Bc4Palette(i32 a, i32 b, i32 palette[8]) noexcept
{
    palette[0] = a;
    palette[1] = b;
    if (a > b)
    {
        for (i32 k = 2; k < 8; ++k)
        {
            palette[k] = ((8 - k) * a + (k - 1) * b + 3) / 7;
        }
    }
    else
    {
        for (i32 k = 2; k < 6; ++k)
        {
            palette[k] = ((6 - k) * a + (k - 1) * b + 2) / 5;
        }
        palette[6] = 0;
        palette[7] = 255;
    }
}

struct Bc4Candidate
{
    i32 a = 0;
    i32 b = 0;
    u8 codes[16] = {};
    u32 error = std::numeric_limits<u32>::max();
};

Bc4Candidate ScoreBc4(const u8* values, i32 a, i32 b) noexcept
{
    Bc4Candidate candidate;
    candidate.a = a;
    candidate.b = b;
    i32 palette[8];
    Bc4Palette(a, b, palette);
    u32 error = 0;
    for (u32 texel = 0; texel < 16; ++texel)
    {
        u32 best = 0;
        u32 bestError = std::numeric_limits<u32>::max();
        for (u32 code = 0; code < 8; ++code)
        {
            const u32 e = Square(palette[code] - values[texel]);
            if (e < bestError)
            {
                bestError = e;
                best = code;
            }
        }
        candidate.codes[texel] = static_cast<u8>(best);
        error += bestError;
    }
    candidate.error = error;
    return candidate;
}

// Weight on endpoint `a` for each code, or a negative value for the fixed
// 0/255 codes of the six-value mode, which do not depend on the endpoints.
f32 Bc4Weight(bool eightValue, u32 code) noexcept
{
    if (code == 0)
    {
        return 1.0f;
    }
    if (code == 1)
    {
        return 0.0f;
    }
    if (eightValue)
    {
        return static_cast<f32>(8 - static_cast<i32>(code)) / 7.0f;
    }
    if (code >= 6)
    {
        return -1.0f;
    }
    return static_cast<f32>(6 - static_cast<i32>(code)) / 5.0f;
}

// Searches one BC4 mode. Eight-value mode needs a > b; six-value mode a <= b
// and gets exact 0 and 255 for free, so its endpoints span only the values
// strictly between.
Bc4Candidate SearchBc4Mode(const u8* values, bool eightValue, u32 passes) noexcept
{
    i32 low = 255;
    i32 high = 0;
    for (u32 texel = 0; texel < 16; ++texel)
    {
        const i32 v = values[texel];
        if (!eightValue && (v == 0 || v == 255))
        {
            continue;
        }
        low = std::min(low, v);
        high = std::max(high, v);
    }
    if (low > high)
    {
        // Six-value mode with only 0 and 255 present: any endpoints work.
        low = 0;
        high = 0;
    }

    const auto order = [eightValue](i32& a, i32& b) noexcept
    {
        if (eightValue)
        {
            if (a < b)
            {
                std::swap(a, b);
            }
            if (a == b)
            {
                // Equal endpoints would select six-value mode; separate them.
                if (a < 255)
                {
                    ++a;
                }
                else
                {
                    --b;
                }
            }
        }
        else if (a > b)
        {
            std::swap(a, b);
        }
    };

    i32 a = eightValue ? high : low;
    i32 b = eightValue ? low : high;
    order(a, b);
    Bc4Candidate best = ScoreBc4(values, a, b);
    for (u32 pass = 0; pass < passes && best.error > 0; ++pass)
    {
        f32 aa = 0.0f;
        f32 ab = 0.0f;
        f32 bb = 0.0f;
        f32 ax = 0.0f;
        f32 bx = 0.0f;
        for (u32 texel = 0; texel < 16; ++texel)
        {
            const f32 w = Bc4Weight(a > b, best.codes[texel]);
            if (w < 0.0f)
            {
                continue;
            }
            const f32 v = 1.0f - w;
            const f32 x = static_cast<f32>(values[texel]);
            aa += w * w;
            ab += w * v;
            bb += v * v;
            ax += w * x;
            bx += v * x;
        }
        const f32 determinant = aa * bb - ab * ab;
        if (std::fabs(determinant) < 1.0e-6f)
        {
            break;
        }
        i32 na = Clamp255(static_cast<i32>(std::lround((bb * ax - ab * bx) / determinant)));
        i32 nb = Clamp255(static_cast<i32>(std::lround((aa * bx - ab * ax) / determinant)));
        order(na, nb);
        const Bc4Candidate refined = ScoreBc4(values, na, nb);
        if (refined.error >= best.error)
        {
            break;
        }
        best = refined;
        a = na;
        b = nb;
    }
    return best;
}

void WriteBc4(Byte* block, const Bc4Candidate& candidate) noexcept
{
    block[0] = static_cast<Byte>(candidate.a);
    block[1] = static_cast<Byte>(candidate.b);
    u64 bits = 0;
    for (u32 texel = 0; texel < 16; ++texel)
    {
        bits |= static_cast<u64>(candidate.codes[texel]) << (texel * 3u);
    }
    for (u32 index = 0; index < 6; ++index)
    {
        block[2 + index] = static_cast<Byte>((bits >> (index * 8u)) & 0xFFu);
    }
}

// ============================================================================
// BC7
// ============================================================================

// Partition and anchor tables from the BC7 (BPTC) format specification.
// Two-subset partitions as 16-bit masks, bit i set when texel i belongs to
// subset 1.
constexpr u16 kPartitions2[64] = {
    0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80, 0xC800, 0xFFEC, 0xFE80,
    0xE800, 0xFFE8, 0xFF00, 0xFFF0, 0xF000, 0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310,
    0x3100, 0x8CCE, 0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C, 0xAAAA,
    0xF0F0, 0x5A5A, 0x33CC, 0x3C3C, 0x55AA, 0x9696, 0xA55A, 0x73CE, 0x13C8, 0x324C, 0x3BDC,
    0x6996, 0xC33C, 0x9966, 0x0660, 0x0272, 0x04E4, 0x4E40, 0x2720, 0xC936, 0x936C, 0x39C6,
    0x639C, 0x9336, 0x9CC6, 0x817E, 0xE718, 0xCCF0, 0x0FCC, 0x7744, 0xEE22,
};

// Three-subset partitions, one subset digit per texel in row-major order.
constexpr const char* kPartitions3[64] = {
    "0011001102210222", "0001001122112221", "0000200122112211", "0222002200110111",
    "0000000011221122", "0011001100220022", "0022002211111111", "0011001122112211",
    "0000000011112222", "0000111111112222", "0000111122222222", "0012001200120012",
    "0112011201120112", "0122012201220122", "0011011211221222", "0011200122002220",
    "0001001101121122", "0111001120012200", "0000112211221122", "0022002200221111",
    "0111011102220222", "0001000122212221", "0000001101220122", "0000110022102210",
    "0122012200110000", "0012001211222222", "0110122112210110", "0000011012211221",
    "0022110211020022", "0110011020022222", "0011012201220011", "0000200022112221",
    "0000000211221222", "0222002200120011", "0011001200220222", "0120012001200120",
    "0000111122220000", "0120120120120120", "0120201212010120", "0011220011220011",
    "0011112222000011", "0101010122222222", "0000000021212121", "0022112200221122",
    "0022001100220011", "0220122102201221", "0101222222220101", "0000212121212121",
    "0101010101012222", "0222011102220111", "0002111200021112", "0000211221122112",
    "0222011101110222", "0002111211120002", "0110011001102222", "0000000021122112",
    "0110011022222222", "0022001100110022", "0022112211220022", "0000000000002112",
    "0002000100020001", "0222122202221222", "0101222222222222", "0111201122012220",
};

constexpr u8 kAnchor2[64] = {
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2,  8, 2,  2, 8,
    8,  15, 2,  8,  2,  2,  8,  8,  2,  2,  15, 15, 6,  8,  2,  8,  15, 15, 2, 8,  2, 2,
    2,  15, 15, 6,  6,  2,  6,  8,  15, 15, 2,  2,  15, 15, 15, 15, 15, 2,  2, 15,
};

constexpr u8 kAnchor3Second[64] = {
    3,  3,  15, 15, 8, 3,  15, 15, 8,  8,  6, 6,  6, 5,  3, 3,  3,  3,  8,  15, 3,  3,
    6,  10, 5,  8,  8, 6,  8,  5,  15, 15, 8, 15, 3, 5,  6, 10, 8,  15, 15, 3,  15, 5,
    15, 15, 15, 15, 3, 15, 5,  5,  5,  8,  5, 10, 5, 10, 8, 13, 15, 12, 3,  3,
};

constexpr u8 kAnchor3Third[64] = {
    15, 8, 8, 3,  15, 15, 3,  8,  15, 15, 15, 15, 15, 15, 15, 8,  15, 8,  15, 3,  15, 8,
    15, 8, 3, 15, 6,  10, 15, 15, 10, 8,  15, 3,  15, 10, 10, 8,  9,  10, 6,  15, 8,  15,
    3,  6, 6, 8,  15, 3,  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3,  15, 15, 8,
};

constexpr u8 kWeights2[4] = {0, 21, 43, 64};
constexpr u8 kWeights3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
constexpr u8 kWeights4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

const u8* WeightsFor(u32 indexBits) noexcept
{
    return indexBits == 2 ? kWeights2 : (indexBits == 3 ? kWeights3 : kWeights4);
}

struct Bc7ModeInfo
{
    u8 subsets;
    u8 partitionBits;
    u8 rotationBits;
    u8 indexSelectionBits;
    u8 colorBits;
    u8 alphaBits;
    u8 endpointPBits;  // one p-bit per endpoint
    u8 sharedPBits;    // one p-bit per subset
    u8 indexBits;
    u8 secondaryIndexBits;
};

constexpr Bc7ModeInfo kBc7Modes[8] = {
    {3, 4, 0, 0, 4, 0, 1, 0, 3, 0}, {2, 6, 0, 0, 6, 0, 0, 1, 3, 0}, {3, 6, 0, 0, 5, 0, 0, 0, 2, 0},
    {2, 6, 0, 0, 7, 0, 1, 0, 2, 0}, {1, 0, 2, 1, 5, 6, 0, 0, 2, 3}, {1, 0, 2, 0, 7, 8, 0, 0, 2, 2},
    {1, 0, 0, 0, 7, 7, 1, 0, 4, 0}, {2, 6, 0, 0, 5, 5, 1, 0, 2, 0},
};

u32 SubsetOf(u32 subsets, u32 partition, u32 texel) noexcept
{
    if (subsets == 2)
    {
        return (kPartitions2[partition] >> texel) & 1u;
    }
    if (subsets == 3)
    {
        return static_cast<u32>(kPartitions3[partition][texel] - '0');
    }
    return 0;
}

bool IsAnchor(u32 subsets, u32 partition, u32 texel) noexcept
{
    if (texel == 0)
    {
        return true;
    }
    if (subsets == 2)
    {
        return texel == kAnchor2[partition];
    }
    if (subsets == 3)
    {
        return texel == kAnchor3Second[partition] || texel == kAnchor3Third[partition];
    }
    return false;
}

u32 AnchorOfSubset(u32 subsets, u32 partition, u32 subset) noexcept
{
    if (subset == 0)
    {
        return 0;
    }
    if (subsets == 2)
    {
        return kAnchor2[partition];
    }
    return subset == 1 ? kAnchor3Second[partition] : kAnchor3Third[partition];
}

// Expands an n-bit endpoint code (n includes any p-bit) to 8 bits by bit
// replication, as the specification defines.
i32 Unquantize(u32 value, u32 bits) noexcept
{
    if (bits >= 8)
    {
        return static_cast<i32>(value);
    }
    value <<= (8u - bits);
    return static_cast<i32>(value | (value >> bits));
}

i32 Interpolate(i32 e0, i32 e1, u32 weight) noexcept
{
    return ((64 - static_cast<i32>(weight)) * e0 + static_cast<i32>(weight) * e1 + 32) >> 6;
}

class BitReader
{
public:
    explicit BitReader(const Byte* block) noexcept
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

class BitWriter
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
    [[nodiscard]] u32 Position() const noexcept
    {
        return m_position;
    }

private:
    Byte m_bytes[16] = {};
    u32 m_position = 0;
};

// Endpoint codes for one BC7 block in the specification's storage order:
// [subset][endpoint][channel], channel 3 is alpha. P-bits per endpoint; modes
// with shared p-bits store the same bit in both endpoints of a subset.
struct Bc7Endpoints
{
    u32 code[3][2][4] = {};
    u32 pbit[3][2] = {};
};

void Bc7UnquantizeEndpoints(const Bc7ModeInfo& mode, const Bc7Endpoints& endpoints, u32 subset,
                            i32 out[2][4]) noexcept
{
    const bool hasPBit = mode.endpointPBits != 0 || mode.sharedPBits != 0;
    for (u32 e = 0; e < 2; ++e)
    {
        for (u32 ch = 0; ch < 4; ++ch)
        {
            const u32 bits = ch < 3 ? mode.colorBits : mode.alphaBits;
            if (ch == 3 && bits == 0)
            {
                out[e][ch] = 255;
                continue;
            }
            u32 value = endpoints.code[subset][e][ch];
            u32 total = bits;
            if (hasPBit)
            {
                value = (value << 1u) | endpoints.pbit[subset][e];
                total += 1;
            }
            out[e][ch] = Unquantize(value, total);
        }
    }
}

// ---- BC7 encoder ------------------------------------------------------------

// What one fit is asked to do: which channels, how many index bits, and how
// endpoints are quantized.
struct Bc7FitSpec
{
    u32 firstChannel = 0;  // 0 for RGB/RGBA fits, 3 for a separate alpha fit
    u32 channels = 3;      // 1, 3 or 4
    u32 bits = 7;          // endpoint bits per channel, excluding p-bits
    u32 alphaBits = 0;     // when channels == 4 and alpha differs (mode 7)
    u32 pbitMode = 0;      // 0 none, 1 per endpoint, 2 shared
    u32 indexBits = 2;
};

struct Bc7FitResult
{
    u32 code[2][4] = {};
    u32 pbit[2] = {};
    u8 index[16] = {};
    u32 error = std::numeric_limits<u32>::max();
};

u32 ChannelBits(const Bc7FitSpec& spec, u32 channel) noexcept
{
    return (channel == 3 && spec.alphaBits != 0) ? spec.alphaBits : spec.bits;
}

// Nearest code for an 8-bit target given a fixed p-bit, and the squared
// reconstruction error of that code.
u32 QuantizeChannel(f32 target, u32 bits, bool hasPBit, u32 pbit, f32& error) noexcept
{
    const u32 total = bits + (hasPBit ? 1u : 0u);
    const u32 maximum = (1u << bits) - 1u;
    const f32 scaled = Clamp255f(target) * static_cast<f32>((1u << total) - 1u) / 255.0f;
    i32 guess =
        static_cast<i32>(hasPBit ? (scaled - static_cast<f32>(pbit)) * 0.5f + 0.5f : scaled + 0.5f);
    u32 best = 0;
    f32 bestError = std::numeric_limits<f32>::max();
    for (i32 candidate = guess - 1; candidate <= guess + 1; ++candidate)
    {
        if (candidate < 0 || candidate > static_cast<i32>(maximum))
        {
            continue;
        }
        const u32 value =
            hasPBit ? ((static_cast<u32>(candidate) << 1u) | pbit) : static_cast<u32>(candidate);
        const f32 diff = static_cast<f32>(Unquantize(value, total)) - target;
        if (diff * diff < bestError)
        {
            bestError = diff * diff;
            best = static_cast<u32>(candidate);
        }
    }
    error = bestError;
    return best;
}

i32 Reconstruct(const Bc7FitSpec& spec, u32 channel, u32 code, u32 pbit) noexcept
{
    const u32 bits = ChannelBits(spec, channel);
    if (spec.pbitMode != 0)
    {
        return Unquantize((code << 1u) | pbit, bits + 1u);
    }
    return Unquantize(code, bits);
}

// Quantizes a float endpoint pair under the spec's p-bit rule. `forced` pins
// the p-bits (for exhaustive p-bit search); otherwise the best are chosen per
// endpoint or per pair.
void QuantizeEndpoints(const Bc7FitSpec& spec, const f32 endpoint[2][4], Bc7FitResult& out,
                       i32 forced0, i32 forced1) noexcept
{
    const bool hasPBit = spec.pbitMode != 0;
    const auto quantizeOne = [&](u32 e, u32 pbit, u32 codes[4]) noexcept
    {
        f32 total = 0.0f;
        for (u32 c = 0; c < spec.channels; ++c)
        {
            const u32 channel = spec.firstChannel + c;
            f32 error = 0.0f;
            codes[channel] = QuantizeChannel(endpoint[e][channel], ChannelBits(spec, channel),
                                             hasPBit, pbit, error);
            total += error;
        }
        return total;
    };

    if (!hasPBit)
    {
        (void)quantizeOne(0, 0, out.code[0]);
        (void)quantizeOne(1, 0, out.code[1]);
        out.pbit[0] = out.pbit[1] = 0;
        return;
    }
    if (spec.pbitMode == 1)
    {
        for (u32 e = 0; e < 2; ++e)
        {
            const i32 forced = e == 0 ? forced0 : forced1;
            if (forced >= 0)
            {
                (void)quantizeOne(e, static_cast<u32>(forced), out.code[e]);
                out.pbit[e] = static_cast<u32>(forced);
                continue;
            }
            u32 codes0[4] = {};
            u32 codes1[4] = {};
            const f32 error0 = quantizeOne(e, 0, codes0);
            const f32 error1 = quantizeOne(e, 1, codes1);
            const bool useOne = error1 < error0;
            std::memcpy(out.code[e], useOne ? codes1 : codes0, sizeof(codes0));
            out.pbit[e] = useOne ? 1u : 0u;
        }
        return;
    }
    // Shared p-bit: both endpoints of the subset use the same bit.
    if (forced0 >= 0)
    {
        (void)quantizeOne(0, static_cast<u32>(forced0), out.code[0]);
        (void)quantizeOne(1, static_cast<u32>(forced0), out.code[1]);
        out.pbit[0] = out.pbit[1] = static_cast<u32>(forced0);
        return;
    }
    u32 a0[4] = {};
    u32 b0[4] = {};
    u32 a1[4] = {};
    u32 b1[4] = {};
    const f32 error0 = quantizeOne(0, 0, a0) + quantizeOne(1, 0, b0);
    const f32 error1 = quantizeOne(0, 1, a1) + quantizeOne(1, 1, b1);
    const bool useOne = error1 < error0;
    std::memcpy(out.code[0], useOne ? a1 : a0, sizeof(a0));
    std::memcpy(out.code[1], useOne ? b1 : b0, sizeof(b0));
    out.pbit[0] = out.pbit[1] = useOne ? 1u : 0u;
}

// Assigns indices against the exact decoded palette and returns the error.
u32 AssignIndices(const Bc7FitSpec& spec, const u8 (*texels)[4], const u8* members, u32 count,
                  Bc7FitResult& fit) noexcept
{
    const u32 entries = 1u << spec.indexBits;
    const u8* weights = WeightsFor(spec.indexBits);
    i32 palette[16][4] = {};
    for (u32 c = 0; c < spec.channels; ++c)
    {
        const u32 channel = spec.firstChannel + c;
        const i32 e0 = Reconstruct(spec, channel, fit.code[0][channel], fit.pbit[0]);
        const i32 e1 = Reconstruct(spec, channel, fit.code[1][channel], fit.pbit[1]);
        for (u32 entry = 0; entry < entries; ++entry)
        {
            palette[entry][channel] = Interpolate(e0, e1, weights[entry]);
        }
    }
    u32 total = 0;
    for (u32 i = 0; i < count; ++i)
    {
        const u8* texel = texels[members[i]];
        u32 best = 0;
        u32 bestError = std::numeric_limits<u32>::max();
        for (u32 entry = 0; entry < entries; ++entry)
        {
            u32 error = 0;
            for (u32 c = 0; c < spec.channels; ++c)
            {
                const u32 channel = spec.firstChannel + c;
                error += Square(palette[entry][channel] - texel[channel]);
            }
            if (error < bestError)
            {
                bestError = error;
                best = entry;
            }
        }
        fit.index[members[i]] = static_cast<u8>(best);
        total += bestError;
    }
    fit.error = total;
    return total;
}

// Fits one subset: line fit for a starting pair, then alternating index
// assignment and least-squares endpoint refits on the exact palette.
Bc7FitResult FitSubset(const Bc7FitSpec& spec, const u8 (*texels)[4], const u8* members, u32 count,
                       BcQuality quality) noexcept
{
    f32 points[16][4] = {};
    for (u32 i = 0; i < count; ++i)
    {
        for (u32 c = 0; c < spec.channels; ++c)
        {
            points[i][c] = static_cast<f32>(texels[members[i]][spec.firstChannel + c]);
        }
    }
    f32 low[4] = {};
    f32 high[4] = {};
    LineFitEndpoints(points, count, spec.channels, low, high);
    f32 endpoint[2][4] = {};
    for (u32 c = 0; c < spec.channels; ++c)
    {
        endpoint[0][spec.firstChannel + c] = low[c];
        endpoint[1][spec.firstChannel + c] = high[c];
    }

    const bool exhaustivePBits = quality == BcQuality::High && spec.pbitMode == 1;
    const auto evaluate = [&](const f32 candidate[2][4]) noexcept
    {
        Bc7FitResult best;
        if (exhaustivePBits)
        {
            for (i32 p = 0; p < 4; ++p)
            {
                Bc7FitResult trial;
                QuantizeEndpoints(spec, candidate, trial, p & 1, p >> 1);
                if (AssignIndices(spec, texels, members, count, trial) < best.error)
                {
                    best = trial;
                }
            }
            return best;
        }
        QuantizeEndpoints(spec, candidate, best, -1, -1);
        (void)AssignIndices(spec, texels, members, count, best);
        return best;
    };

    Bc7FitResult best = evaluate(endpoint);
    const u8* weights = WeightsFor(spec.indexBits);
    const u32 passes = RefinementPasses(quality);
    Bc7FitResult current = best;
    for (u32 pass = 0; pass < passes && best.error > 0; ++pass)
    {
        f32 aa = 0.0f;
        f32 ab = 0.0f;
        f32 bb = 0.0f;
        f32 ax[4] = {};
        f32 bx[4] = {};
        for (u32 i = 0; i < count; ++i)
        {
            const f32 w = static_cast<f32>(weights[current.index[members[i]]]) / 64.0f;
            const f32 v = 1.0f - w;
            aa += v * v;
            ab += v * w;
            bb += w * w;
            for (u32 c = 0; c < spec.channels; ++c)
            {
                const f32 x = static_cast<f32>(texels[members[i]][spec.firstChannel + c]);
                ax[c] += v * x;
                bx[c] += w * x;
            }
        }
        const f32 determinant = aa * bb - ab * ab;
        if (std::fabs(determinant) < 1.0e-6f)
        {
            break;
        }
        f32 refined[2][4] = {};
        for (u32 c = 0; c < spec.channels; ++c)
        {
            refined[0][spec.firstChannel + c] = Clamp255f((bb * ax[c] - ab * bx[c]) / determinant);
            refined[1][spec.firstChannel + c] = Clamp255f((aa * bx[c] - ab * ax[c]) / determinant);
        }
        current = evaluate(refined);
        if (current.error < best.error)
        {
            best = current;
        }
        else
        {
            break;
        }
    }
    return best;
}

struct Bc7Block
{
    u32 mode = 6;
    u32 partition = 0;
    u32 rotation = 0;        // mode 4 and 5: which channel swaps with alpha
    u32 indexSelection = 0;  // mode 4: 1 gives the colour the 3-bit index set
    Bc7Endpoints endpoints;
    u8 index[16] = {};
    u8 secondary[16] = {};
    u32 error = std::numeric_limits<u32>::max();
};

// Writes a candidate after enforcing the anchor rule: each subset's anchor
// texel must have its index MSB clear, which is fixed by swapping that
// subset's endpoints and mirroring its indices.
void WriteBc7(const Bc7Block& source, Byte* out) noexcept
{
    Bc7Block block = source;
    const Bc7ModeInfo& mode = kBc7Modes[block.mode];
    const u32 indexBits = mode.indexBits;
    const u32 highBit = 1u << (indexBits - 1u);
    const u32 maximum = (1u << indexBits) - 1u;
    const u32 colorChannels = mode.secondaryIndexBits != 0 ? 3u : 4u;
    // The primary index set drives the colour channels, except in mode 4 with
    // index selection, where it drives alpha and the secondary set the colour.
    const bool swapRoles = mode.secondaryIndexBits != 0 && block.indexSelection != 0;
    const u32 primaryFirst = swapRoles ? 3u : 0u;
    const u32 primaryEnd = swapRoles ? 4u : colorChannels;
    for (u32 subset = 0; subset < mode.subsets; ++subset)
    {
        const u32 anchor = AnchorOfSubset(mode.subsets, block.partition, subset);
        if ((block.index[anchor] & highBit) == 0)
        {
            continue;
        }
        for (u32 ch = primaryFirst; ch < primaryEnd; ++ch)
        {
            std::swap(block.endpoints.code[subset][0][ch], block.endpoints.code[subset][1][ch]);
        }
        if (mode.endpointPBits != 0)
        {
            std::swap(block.endpoints.pbit[subset][0], block.endpoints.pbit[subset][1]);
        }
        for (u32 texel = 0; texel < 16; ++texel)
        {
            if (SubsetOf(mode.subsets, block.partition, texel) == subset)
            {
                block.index[texel] = static_cast<u8>(maximum - block.index[texel]);
            }
        }
    }
    if (mode.secondaryIndexBits != 0)
    {
        const u32 secondaryHigh = 1u << (mode.secondaryIndexBits - 1u);
        const u32 secondaryMaximum = (1u << mode.secondaryIndexBits) - 1u;
        if ((block.secondary[0] & secondaryHigh) != 0)
        {
            const u32 secondaryFirst = swapRoles ? 0u : 3u;
            const u32 secondaryEnd = swapRoles ? 3u : 4u;
            for (u32 ch = secondaryFirst; ch < secondaryEnd; ++ch)
            {
                std::swap(block.endpoints.code[0][0][ch], block.endpoints.code[0][1][ch]);
            }
            for (u32 texel = 0; texel < 16; ++texel)
            {
                block.secondary[texel] = static_cast<u8>(secondaryMaximum - block.secondary[texel]);
            }
        }
    }

    BitWriter writer;
    writer.Write(1u << block.mode, block.mode + 1u);
    writer.Write(block.partition, mode.partitionBits);
    writer.Write(block.rotation, mode.rotationBits);
    writer.Write(block.indexSelection, mode.indexSelectionBits);
    for (u32 ch = 0; ch < 3; ++ch)
    {
        for (u32 subset = 0; subset < mode.subsets; ++subset)
        {
            writer.Write(block.endpoints.code[subset][0][ch], mode.colorBits);
            writer.Write(block.endpoints.code[subset][1][ch], mode.colorBits);
        }
    }
    if (mode.alphaBits != 0)
    {
        for (u32 subset = 0; subset < mode.subsets; ++subset)
        {
            writer.Write(block.endpoints.code[subset][0][3], mode.alphaBits);
            writer.Write(block.endpoints.code[subset][1][3], mode.alphaBits);
        }
    }
    if (mode.endpointPBits != 0)
    {
        for (u32 subset = 0; subset < mode.subsets; ++subset)
        {
            writer.Write(block.endpoints.pbit[subset][0], 1);
            writer.Write(block.endpoints.pbit[subset][1], 1);
        }
    }
    if (mode.sharedPBits != 0)
    {
        for (u32 subset = 0; subset < mode.subsets; ++subset)
        {
            writer.Write(block.endpoints.pbit[subset][0], 1);
        }
    }
    for (u32 texel = 0; texel < 16; ++texel)
    {
        const bool anchor = IsAnchor(mode.subsets, block.partition, texel);
        writer.Write(block.index[texel], indexBits - (anchor ? 1u : 0u));
    }
    if (mode.secondaryIndexBits != 0)
    {
        for (u32 texel = 0; texel < 16; ++texel)
        {
            writer.Write(block.secondary[texel], mode.secondaryIndexBits - (texel == 0 ? 1u : 0u));
        }
    }
    writer.CopyTo(out);
}

// Encodes a partitioned or single-subset mode where all channels share one
// index set (modes 0, 1, 3, 6).
Bc7Block EncodeBc7Joint(const u8 (*texels)[4], u32 modeIndex, u32 partition,
                        BcQuality quality) noexcept
{
    const Bc7ModeInfo& mode = kBc7Modes[modeIndex];
    Bc7FitSpec spec;
    spec.firstChannel = 0;
    spec.channels = mode.alphaBits != 0 ? 4u : 3u;
    spec.bits = mode.colorBits;
    spec.alphaBits = mode.alphaBits;
    spec.pbitMode = mode.endpointPBits != 0 ? 1u : (mode.sharedPBits != 0 ? 2u : 0u);
    spec.indexBits = mode.indexBits;

    Bc7Block block;
    block.mode = modeIndex;
    block.partition = partition;
    block.error = 0;
    for (u32 subset = 0; subset < mode.subsets; ++subset)
    {
        u8 members[16] = {};
        u32 count = 0;
        for (u32 texel = 0; texel < 16; ++texel)
        {
            if (SubsetOf(mode.subsets, partition, texel) == subset)
            {
                members[count++] = static_cast<u8>(texel);
            }
        }
        const Bc7FitResult fit = FitSubset(spec, texels, members, count, quality);
        for (u32 e = 0; e < 2; ++e)
        {
            for (u32 ch = 0; ch < 4; ++ch)
            {
                block.endpoints.code[subset][e][ch] = fit.code[e][ch];
            }
            block.endpoints.pbit[subset][e] = fit.pbit[e];
        }
        for (u32 i = 0; i < count; ++i)
        {
            block.index[members[i]] = fit.index[members[i]];
        }
        block.error += fit.error;
    }
    if (mode.alphaBits == 0)
    {
        // Colour-only modes decode alpha as 255.
        for (u32 texel = 0; texel < 16; ++texel)
        {
            block.error += Square(255 - texels[texel][3]);
        }
    }
    return block;
}

// Mode 5: RGB and alpha fitted separately, each with its own 2-bit indices, and
// (at High) a channel rotation that lets any one of R, G, B swap places with
// alpha, which helps when alpha is smooth and one colour channel is not. Every
// rotation decodes to a permutation of the same four channels, so the squared
// error measured on the rotated texels is the error of the real ones.
Bc7Block EncodeBc7Mode5(const u8 (*texels)[4], BcQuality quality) noexcept
{
    u8 members[16];
    for (u32 texel = 0; texel < 16; ++texel)
    {
        members[texel] = static_cast<u8>(texel);
    }
    const u32 rotations = quality == BcQuality::High ? 4u : 1u;

    Bc7Block best;
    best.mode = 5;
    for (u32 rotation = 0; rotation < rotations; ++rotation)
    {
        u8 rotated[16][4];
        for (u32 texel = 0; texel < 16; ++texel)
        {
            for (u32 ch = 0; ch < 4; ++ch)
            {
                rotated[texel][ch] = texels[texel][ch];
            }
            if (rotation != 0)
            {
                std::swap(rotated[texel][rotation - 1u], rotated[texel][3]);
            }
        }
        Bc7FitSpec color;
        color.channels = 3;
        color.bits = 7;
        color.indexBits = 2;
        Bc7FitSpec alpha;
        alpha.firstChannel = 3;
        alpha.channels = 1;
        alpha.bits = 8;
        alpha.indexBits = 2;
        const Bc7FitResult colorFit = FitSubset(color, rotated, members, 16, quality);
        const Bc7FitResult alphaFit = FitSubset(alpha, rotated, members, 16, quality);
        const u32 error = colorFit.error + alphaFit.error;
        if (error >= best.error)
        {
            continue;
        }
        best.rotation = rotation;
        for (u32 e = 0; e < 2; ++e)
        {
            for (u32 ch = 0; ch < 3; ++ch)
            {
                best.endpoints.code[0][e][ch] = colorFit.code[e][ch];
            }
            best.endpoints.code[0][e][3] = alphaFit.code[e][3];
        }
        for (u32 texel = 0; texel < 16; ++texel)
        {
            best.index[texel] = colorFit.index[texel];
            best.secondary[texel] = alphaFit.index[texel];
        }
        best.error = error;
    }
    return best;
}

// Mode 4: colour (5 bits) and alpha (6 bits) fitted separately like mode 5, but
// with one set of 2-bit and one of 3-bit indices that `indexSelection` assigns
// to colour or alpha, and a channel rotation that lets any one of R, G, B swap
// places with alpha. The better of the 2 x 4 combinations is kept; every
// combination decodes to a permutation of the same four channels, so the
// squared error measured on the rotated texels is the error of the real ones.
Bc7Block EncodeBc7Mode4(const u8 (*texels)[4], BcQuality quality) noexcept
{
    u8 members[16];
    for (u32 texel = 0; texel < 16; ++texel)
    {
        members[texel] = static_cast<u8>(texel);
    }
    const u32 rotations = quality == BcQuality::High ? 4u : 1u;

    Bc7Block best;
    best.mode = 4;
    for (u32 rotation = 0; rotation < rotations; ++rotation)
    {
        u8 rotated[16][4];
        for (u32 texel = 0; texel < 16; ++texel)
        {
            for (u32 ch = 0; ch < 4; ++ch)
            {
                rotated[texel][ch] = texels[texel][ch];
            }
            if (rotation != 0)
            {
                std::swap(rotated[texel][rotation - 1u], rotated[texel][3]);
            }
        }
        for (u32 selection = 0; selection < 2; ++selection)
        {
            Bc7FitSpec color;
            color.channels = 3;
            color.bits = 5;
            color.indexBits = selection != 0 ? 3u : 2u;
            Bc7FitSpec alpha;
            alpha.firstChannel = 3;
            alpha.channels = 1;
            alpha.bits = 6;
            alpha.indexBits = selection != 0 ? 2u : 3u;
            const Bc7FitResult colorFit = FitSubset(color, rotated, members, 16, quality);
            const Bc7FitResult alphaFit = FitSubset(alpha, rotated, members, 16, quality);
            const u32 error = colorFit.error + alphaFit.error;
            if (error >= best.error)
            {
                continue;
            }
            best.rotation = rotation;
            best.indexSelection = selection;
            for (u32 e = 0; e < 2; ++e)
            {
                for (u32 ch = 0; ch < 3; ++ch)
                {
                    best.endpoints.code[0][e][ch] = colorFit.code[e][ch];
                }
                best.endpoints.code[0][e][3] = alphaFit.code[e][3];
            }
            for (u32 texel = 0; texel < 16; ++texel)
            {
                best.index[texel] = selection != 0 ? alphaFit.index[texel] : colorFit.index[texel];
                best.secondary[texel] =
                    selection != 0 ? colorFit.index[texel] : alphaFit.index[texel];
            }
            best.error = error;
        }
    }
    return best;
}

// Optimal single-colour pairs for BC7 mode 5: for each 8-bit value, the two
// 7-bit endpoints whose 2-bit index-1 interpolant (weight 21/64) reproduces it.
// Every value is reachable, so solid blocks round-trip exactly; mode 6 cannot
// promise that because its p-bit is shared by all channels of an endpoint.
struct Bc7SingleColorTable
{
    u8 pair[256][2] = {};

    Bc7SingleColorTable() noexcept
    {
        for (i32 value = 0; value < 256; ++value)
        {
            i32 bestError = 1 << 30;
            for (u32 a = 0; a < 128; ++a)
            {
                for (u32 b = 0; b < 128; ++b)
                {
                    const i32 decoded =
                        Interpolate(Unquantize(a, 7), Unquantize(b, 7), kWeights2[1]);
                    const i32 error = std::abs(decoded - value);
                    if (error < bestError)
                    {
                        bestError = error;
                        pair[value][0] = static_cast<u8>(a);
                        pair[value][1] = static_cast<u8>(b);
                    }
                }
            }
        }
    }
};

bool EncodeBc7Solid(const u8 (*texels)[4], Bc7Block& out) noexcept
{
    for (u32 texel = 1; texel < 16; ++texel)
    {
        if (std::memcmp(texels[texel], texels[0], 4) != 0)
        {
            return false;
        }
    }
    static const Bc7SingleColorTable table{};
    out = Bc7Block{};
    out.mode = 5;
    for (u32 ch = 0; ch < 3; ++ch)
    {
        out.endpoints.code[0][0][ch] = table.pair[texels[0][ch]][0];
        out.endpoints.code[0][1][ch] = table.pair[texels[0][ch]][1];
    }
    out.endpoints.code[0][0][3] = texels[0][3];
    out.endpoints.code[0][1][3] = texels[0][3];
    u32 error = 0;
    for (u32 ch = 0; ch < 3; ++ch)
    {
        const i32 decoded = Interpolate(Unquantize(out.endpoints.code[0][0][ch], 7),
                                        Unquantize(out.endpoints.code[0][1][ch], 7), kWeights2[1]);
        error += Square(decoded - texels[0][ch]) * 16u;
    }
    for (u32 texel = 0; texel < 16; ++texel)
    {
        out.index[texel] = 1;
        out.secondary[texel] = 0;
    }
    out.error = error;
    return true;
}

// Ranks partitions by the summed line-fit residual of their subsets and
// returns the best `wanted` of the first `available`.
u32 RankPartitions(const u8 (*texels)[4], u32 subsets, u32 available, u32 wanted, u32* out) noexcept
{
    f32 scores[64];
    u32 order[64];
    for (u32 partition = 0; partition < available; ++partition)
    {
        f32 score = 0.0f;
        for (u32 subset = 0; subset < subsets; ++subset)
        {
            f32 points[16][4] = {};
            u32 count = 0;
            for (u32 texel = 0; texel < 16; ++texel)
            {
                if (SubsetOf(subsets, partition, texel) == subset)
                {
                    for (u32 ch = 0; ch < 3; ++ch)
                    {
                        points[count][ch] = static_cast<f32>(texels[texel][ch]);
                    }
                    ++count;
                }
            }
            score += LineFitResidual(points, count, 3);
        }
        scores[partition] = score;
        order[partition] = partition;
    }
    const u32 count = std::min(wanted, available);
    std::partial_sort(
        order, order + count, order + available, [&scores](u32 lhs, u32 rhs) noexcept
        { return scores[lhs] < scores[rhs] || (scores[lhs] == scores[rhs] && lhs < rhs); });
    for (u32 i = 0; i < count; ++i)
    {
        out[i] = order[i];
    }
    return count;
}

}  // namespace

// ============================================================================
// Public block API
// ============================================================================

void DecodeBc1Block(const Byte* block, u8* rgba) noexcept
{
    const u32 c0 = static_cast<u32>(block[0]) | (static_cast<u32>(block[1]) << 8u);
    const u32 c1 = static_cast<u32>(block[2]) | (static_cast<u32>(block[3]) << 8u);
    const u32 indices = ReadLe32(block + 4);
    i32 palette[4][4];
    Bc1Palette(c0, c1, palette);
    for (u32 texel = 0; texel < 16; ++texel)
    {
        const u32 index = (indices >> (texel * 2u)) & 3u;
        for (u32 ch = 0; ch < 4; ++ch)
        {
            rgba[texel * 4 + ch] = static_cast<u8>(palette[index][ch]);
        }
    }
}

void DecodeBc4Block(const Byte* block, u8* red) noexcept
{
    i32 palette[8];
    Bc4Palette(static_cast<i32>(block[0]), static_cast<i32>(block[1]), palette);
    u64 bits = 0;
    for (u32 index = 0; index < 6; ++index)
    {
        bits |= static_cast<u64>(block[2 + index]) << (index * 8u);
    }
    for (u32 texel = 0; texel < 16; ++texel)
    {
        red[texel] = static_cast<u8>(palette[(bits >> (texel * 3u)) & 7u]);
    }
}

void DecodeBc5Block(const Byte* block, u8* redGreen) noexcept
{
    u8 red[16];
    u8 green[16];
    DecodeBc4Block(block, red);
    DecodeBc4Block(block + 8, green);
    for (u32 texel = 0; texel < 16; ++texel)
    {
        redGreen[texel * 2] = red[texel];
        redGreen[texel * 2 + 1] = green[texel];
    }
}

void DecodeBc7Block(const Byte* block, u8* rgba) noexcept
{
    BitReader reader(block);
    u32 modeIndex = 0;
    while (modeIndex < 8 && reader.Read(1) == 0)
    {
        ++modeIndex;
    }
    if (modeIndex >= 8)
    {
        std::memset(rgba, 0, kBcRgbaBlockBytes);
        return;
    }
    const Bc7ModeInfo& mode = kBc7Modes[modeIndex];
    const u32 partition = reader.Read(mode.partitionBits);
    const u32 rotation = reader.Read(mode.rotationBits);
    const u32 indexSelection = reader.Read(mode.indexSelectionBits);

    Bc7Endpoints endpoints;
    for (u32 ch = 0; ch < 3; ++ch)
    {
        for (u32 subset = 0; subset < mode.subsets; ++subset)
        {
            endpoints.code[subset][0][ch] = reader.Read(mode.colorBits);
            endpoints.code[subset][1][ch] = reader.Read(mode.colorBits);
        }
    }
    if (mode.alphaBits != 0)
    {
        for (u32 subset = 0; subset < mode.subsets; ++subset)
        {
            endpoints.code[subset][0][3] = reader.Read(mode.alphaBits);
            endpoints.code[subset][1][3] = reader.Read(mode.alphaBits);
        }
    }
    if (mode.endpointPBits != 0)
    {
        for (u32 subset = 0; subset < mode.subsets; ++subset)
        {
            endpoints.pbit[subset][0] = reader.Read(1);
            endpoints.pbit[subset][1] = reader.Read(1);
        }
    }
    if (mode.sharedPBits != 0)
    {
        for (u32 subset = 0; subset < mode.subsets; ++subset)
        {
            const u32 bit = reader.Read(1);
            endpoints.pbit[subset][0] = bit;
            endpoints.pbit[subset][1] = bit;
        }
    }

    u32 primary[16] = {};
    u32 secondary[16] = {};
    for (u32 texel = 0; texel < 16; ++texel)
    {
        const bool anchor = IsAnchor(mode.subsets, partition, texel);
        primary[texel] = reader.Read(mode.indexBits - (anchor ? 1u : 0u));
    }
    if (mode.secondaryIndexBits != 0)
    {
        for (u32 texel = 0; texel < 16; ++texel)
        {
            secondary[texel] = reader.Read(mode.secondaryIndexBits - (texel == 0 ? 1u : 0u));
        }
    }

    i32 unquantized[3][2][4] = {};
    for (u32 subset = 0; subset < mode.subsets; ++subset)
    {
        Bc7UnquantizeEndpoints(mode, endpoints, subset, unquantized[subset]);
    }

    for (u32 texel = 0; texel < 16; ++texel)
    {
        const u32 subset = SubsetOf(mode.subsets, partition, texel);
        const i32(&e)[2][4] = unquantized[subset];
        u32 colorWeight = 0;
        u32 alphaWeight = 0;
        if (mode.secondaryIndexBits == 0)
        {
            colorWeight = WeightsFor(mode.indexBits)[primary[texel]];
            alphaWeight = colorWeight;
        }
        else if (indexSelection == 0)
        {
            colorWeight = WeightsFor(mode.indexBits)[primary[texel]];
            alphaWeight = WeightsFor(mode.secondaryIndexBits)[secondary[texel]];
        }
        else
        {
            colorWeight = WeightsFor(mode.secondaryIndexBits)[secondary[texel]];
            alphaWeight = WeightsFor(mode.indexBits)[primary[texel]];
        }
        i32 out[4];
        for (u32 ch = 0; ch < 3; ++ch)
        {
            out[ch] = Interpolate(e[0][ch], e[1][ch], colorWeight);
        }
        out[3] = Interpolate(e[0][3], e[1][3], alphaWeight);
        switch (rotation)
        {
        case 1:
            std::swap(out[0], out[3]);
            break;
        case 2:
            std::swap(out[1], out[3]);
            break;
        case 3:
            std::swap(out[2], out[3]);
            break;
        default:
            break;
        }
        for (u32 ch = 0; ch < 4; ++ch)
        {
            rgba[texel * 4 + ch] = static_cast<u8>(out[ch]);
        }
    }
}

void EncodeBc1Block(const u8* rgba, Byte* block, const BcEncodeSettings& settings) noexcept
{
    bool transparent[16];
    u32 opaqueCount = 0;
    for (u32 texel = 0; texel < 16; ++texel)
    {
        transparent[texel] =
            settings.bc1AlphaThreshold != 0 && rgba[texel * 4 + 3] < settings.bc1AlphaThreshold;
        opaqueCount += transparent[texel] ? 0u : 1u;
    }
    const bool hasTransparency = opaqueCount < 16;
    if (opaqueCount == 0)
    {
        Bc1Candidate candidate;
        candidate.c0 = 0;
        candidate.c1 = 0;
        candidate.indices = 0xFFFFFFFFu;
        WriteBc1(block, candidate);
        return;
    }

    // Solid opaque blocks: the optimal single-colour tables are usually exact.
    bool solid = !hasTransparency;
    for (u32 texel = 1; texel < 16 && solid; ++texel)
    {
        solid = rgba[texel * 4] == rgba[0] && rgba[texel * 4 + 1] == rgba[1] &&
                rgba[texel * 4 + 2] == rgba[2];
    }

    Bc1Candidate best;
    if (solid)
    {
        const SingleColorTables& tables = GetSingleColorTables();
        const Rgb565 a{tables.match5[rgba[0]][0], tables.match6[rgba[1]][0],
                       tables.match5[rgba[2]][0]};
        const Rgb565 b{tables.match5[rgba[0]][1], tables.match6[rgba[1]][1],
                       tables.match5[rgba[2]][1]};
        best = ScoreBc1Mode(rgba, transparent, a, b, true);
    }

    f32 points[16][4] = {};
    u32 count = 0;
    for (u32 texel = 0; texel < 16; ++texel)
    {
        if (!transparent[texel])
        {
            for (u32 ch = 0; ch < 3; ++ch)
            {
                points[count][ch] = static_cast<f32>(rgba[texel * 4 + ch]);
            }
            ++count;
        }
    }
    f32 low[4] = {};
    f32 high[4] = {};
    LineFitEndpoints(points, count, 3, low, high);

    const u32 passes = RefinementPasses(settings.quality);
    const auto search = [&](bool fourColor) noexcept
    {
        Bc1Candidate local =
            ScoreBc1Mode(rgba, transparent, Quantize565(high), Quantize565(low), fourColor);
        for (u32 pass = 0; pass < passes && local.error > 0; ++pass)
        {
            f32 e0[3];
            f32 e1[3];
            if (!RefitBc1(rgba, transparent, local, e0, e1))
            {
                break;
            }
            const Bc1Candidate refined =
                ScoreBc1Mode(rgba, transparent, Quantize565(e0), Quantize565(e1), fourColor);
            if (refined.error >= local.error)
            {
                break;
            }
            local = refined;
        }
        return local;
    };

    if (!hasTransparency)
    {
        const Bc1Candidate four = search(true);
        if (four.error < best.error)
        {
            best = four;
        }
    }
    if (hasTransparency || settings.quality != BcQuality::Fast)
    {
        const Bc1Candidate three = search(false);
        if (three.error < best.error)
        {
            best = three;
        }
    }
    WriteBc1(block, best);
}

void EncodeBc4Block(const u8* red, Byte* block, const BcEncodeSettings& settings) noexcept
{
    const u32 passes = RefinementPasses(settings.quality);
    Bc4Candidate best = SearchBc4Mode(red, true, passes);
    if (best.error > 0)
    {
        const Bc4Candidate six = SearchBc4Mode(red, false, passes);
        if (six.error < best.error)
        {
            best = six;
        }
    }
    WriteBc4(block, best);
}

void EncodeBc5Block(const u8* redGreen, Byte* block, const BcEncodeSettings& settings) noexcept
{
    u8 red[16];
    u8 green[16];
    for (u32 texel = 0; texel < 16; ++texel)
    {
        red[texel] = redGreen[texel * 2];
        green[texel] = redGreen[texel * 2 + 1];
    }
    EncodeBc4Block(red, block, settings);
    EncodeBc4Block(green, block + 8, settings);
}

// ---- BC2 and BC3 --------------------------------------------------------------
//
// Both are 16-byte blocks: eight bytes of alpha (BC2: sixteen explicit 4-bit
// values; BC3: a BC4 block) followed by an eight-byte colour block. Unlike BC1,
// the colour block is always decoded in four-colour mode, whatever the order of
// its two endpoints, so the encoder scores indices against that palette.

namespace
{

void Bc1FourColorPalette(u32 c0, u32 c1, i32 palette[4][3]) noexcept
{
    const Rgb565 a = Unpack565(c0);
    const Rgb565 b = Unpack565(c1);
    const i32 e0[3] = {Expand5(a.r), Expand6(a.g), Expand5(a.b)};
    const i32 e1[3] = {Expand5(b.r), Expand6(b.g), Expand5(b.b)};
    for (u32 ch = 0; ch < 3; ++ch)
    {
        palette[0][ch] = e0[ch];
        palette[1][ch] = e1[ch];
        palette[2][ch] = (2 * e0[ch] + e1[ch] + 1) / 3;
        palette[3][ch] = (e0[ch] + 2 * e1[ch] + 1) / 3;
    }
}

void DecodeFourColorBlock(const Byte* block, u8* rgba) noexcept
{
    const u32 c0 = static_cast<u32>(block[0]) | (static_cast<u32>(block[1]) << 8u);
    const u32 c1 = static_cast<u32>(block[2]) | (static_cast<u32>(block[3]) << 8u);
    const u32 indices = ReadLe32(block + 4);
    i32 palette[4][3];
    Bc1FourColorPalette(c0, c1, palette);
    for (u32 texel = 0; texel < 16; ++texel)
    {
        const u32 index = (indices >> (texel * 2u)) & 3u;
        for (u32 ch = 0; ch < 3; ++ch)
        {
            rgba[texel * 4 + ch] = static_cast<u8>(palette[index][ch]);
        }
    }
}

// Encodes the colour of `rgba` as an opaque BC1 block and then re-picks every
// index against the four-colour palette of the chosen endpoints, so a block the
// BC1 encoder emitted in three-colour mode still decodes to what was scored.
void EncodeFourColorBlock(const u8* rgba, Byte* block, const BcEncodeSettings& settings) noexcept
{
    u8 opaque[kBcRgbaBlockBytes];
    for (u32 texel = 0; texel < 16; ++texel)
    {
        opaque[texel * 4] = rgba[texel * 4];
        opaque[texel * 4 + 1] = rgba[texel * 4 + 1];
        opaque[texel * 4 + 2] = rgba[texel * 4 + 2];
        opaque[texel * 4 + 3] = 255;
    }
    BcEncodeSettings colorSettings = settings;
    colorSettings.bc1AlphaThreshold = 0;
    EncodeBc1Block(opaque, block, colorSettings);
    const u32 c0 = static_cast<u32>(block[0]) | (static_cast<u32>(block[1]) << 8u);
    const u32 c1 = static_cast<u32>(block[2]) | (static_cast<u32>(block[3]) << 8u);
    i32 palette[4][3];
    Bc1FourColorPalette(c0, c1, palette);
    u32 indices = 0;
    for (u32 texel = 0; texel < 16; ++texel)
    {
        u32 bestIndex = 0;
        i32 bestError = std::numeric_limits<i32>::max();
        for (u32 index = 0; index < 4; ++index)
        {
            i32 error = 0;
            for (u32 ch = 0; ch < 3; ++ch)
            {
                const i32 diff = palette[index][ch] - static_cast<i32>(rgba[texel * 4 + ch]);
                error += diff * diff;
            }
            if (error < bestError)
            {
                bestError = error;
                bestIndex = index;
            }
        }
        indices |= bestIndex << (texel * 2u);
    }
    for (u32 byte = 0; byte < 4; ++byte)
    {
        block[4 + byte] = static_cast<Byte>((indices >> (byte * 8u)) & 0xFFu);
    }
}

}  // namespace

void DecodeBc2Block(const Byte* block, u8* rgba) noexcept
{
    DecodeFourColorBlock(block + 8, rgba);
    for (u32 texel = 0; texel < 16; ++texel)
    {
        const u32 byte = static_cast<u32>(block[texel / 2]);
        const u32 nibble = (texel & 1u) != 0 ? (byte >> 4u) : (byte & 15u);
        rgba[texel * 4 + 3] = static_cast<u8>(nibble * 17u);
    }
}

void DecodeBc3Block(const Byte* block, u8* rgba) noexcept
{
    DecodeFourColorBlock(block + 8, rgba);
    u8 alpha[16];
    DecodeBc4Block(block, alpha);
    for (u32 texel = 0; texel < 16; ++texel)
    {
        rgba[texel * 4 + 3] = alpha[texel];
    }
}

void EncodeBc2Block(const u8* rgba, Byte* block, const BcEncodeSettings& settings) noexcept
{
    for (u32 pair = 0; pair < 8; ++pair)
    {
        const u32 low = (static_cast<u32>(rgba[(pair * 2) * 4 + 3]) * 15u + 127u) / 255u;
        const u32 high = (static_cast<u32>(rgba[(pair * 2 + 1) * 4 + 3]) * 15u + 127u) / 255u;
        block[pair] = static_cast<Byte>(low | (high << 4u));
    }
    EncodeFourColorBlock(rgba, block + 8, settings);
}

void EncodeBc3Block(const u8* rgba, Byte* block, const BcEncodeSettings& settings) noexcept
{
    u8 alpha[16];
    for (u32 texel = 0; texel < 16; ++texel)
    {
        alpha[texel] = rgba[texel * 4 + 3];
    }
    EncodeBc4Block(alpha, block, settings);
    EncodeFourColorBlock(rgba, block + 8, settings);
}

void EncodeBc7Block(const u8* rgba, Byte* block, const BcEncodeSettings& settings) noexcept
{
    u8 texels[16][4];
    bool opaque = true;
    for (u32 texel = 0; texel < 16; ++texel)
    {
        for (u32 ch = 0; ch < 4; ++ch)
        {
            texels[texel][ch] = rgba[texel * 4 + ch];
        }
        opaque = opaque && texels[texel][3] == 255;
    }

    Bc7Block solid;
    if (EncodeBc7Solid(texels, solid) && solid.error == 0)
    {
        WriteBc7(solid, block);
        return;
    }

    Bc7Block best = EncodeBc7Joint(texels, 6, 0, settings.quality);
    if (best.error == 0)
    {
        WriteBc7(best, block);
        return;
    }

    if (!opaque)
    {
        const Bc7Block mode5 = EncodeBc7Mode5(texels, settings.quality);
        if (mode5.error < best.error)
        {
            best = mode5;
        }
        if (settings.quality != BcQuality::Fast && best.error > 0)
        {
            const Bc7Block mode4 = EncodeBc7Mode4(texels, settings.quality);
            if (mode4.error < best.error)
            {
                best = mode4;
            }
        }
        // Mode 7: two RGBA subsets, for blocks whose colour and alpha both
        // split into two groups (an alpha edge across a colour edge).
        const u32 candidates7 = settings.quality == BcQuality::Fast
                                    ? 0u
                                    : (settings.quality == BcQuality::Normal ? 3u : 8u);
        if (candidates7 > 0 && best.error > 0)
        {
            u32 partitions7[64];
            const u32 count7 = RankPartitions(texels, 2, 64, candidates7, partitions7);
            for (u32 i = 0; i < count7 && best.error > 0; ++i)
            {
                const Bc7Block trial = EncodeBc7Joint(texels, 7, partitions7[i], settings.quality);
                if (trial.error < best.error)
                {
                    best = trial;
                }
            }
        }
        WriteBc7(best, block);
        return;
    }

    u32 candidates2 = 0;
    u32 candidates3 = 0;
    switch (settings.quality)
    {
    case BcQuality::Fast:
        candidates2 = 1;
        break;
    case BcQuality::Normal:
        candidates2 = 4;
        break;
    case BcQuality::High:
        candidates2 = 12;
        candidates3 = 4;
        break;
    }

    u32 partitions[64];
    const u32 count2 = RankPartitions(texels, 2, 64, candidates2, partitions);
    for (u32 i = 0; i < count2 && best.error > 0; ++i)
    {
        for (const u32 modeIndex : {1u, 3u})
        {
            const Bc7Block trial =
                EncodeBc7Joint(texels, modeIndex, partitions[i], settings.quality);
            if (trial.error < best.error)
            {
                best = trial;
            }
        }
    }
    if (candidates3 > 0 && best.error > 0)
    {
        // Mode 0 addresses only the first 16 three-subset partitions.
        const u32 count3 = RankPartitions(texels, 3, 16, candidates3, partitions);
        for (u32 i = 0; i < count3; ++i)
        {
            const Bc7Block trial = EncodeBc7Joint(texels, 0, partitions[i], settings.quality);
            if (trial.error < best.error)
            {
                best = trial;
            }
        }
        // Mode 2: three subsets with 5-bit endpoints and no p-bits, over all
        // 64 three-subset partitions.
        if (best.error > 0)
        {
            const u32 count3Wide = RankPartitions(texels, 3, 64, candidates3, partitions);
            for (u32 i = 0; i < count3Wide && best.error > 0; ++i)
            {
                const Bc7Block trial = EncodeBc7Joint(texels, 2, partitions[i], settings.quality);
                if (trial.error < best.error)
                {
                    best = trial;
                }
            }
        }
    }
    WriteBc7(best, block);
}

// ============================================================================
// Surfaces
// ============================================================================

namespace
{

ImportStatus BuildLevelTable(TextureAsset& texture) noexcept
{
    if (!texture.levels.TryResize(texture.mipCount))
    {
        return ImportStatus::Err(MakeError(ErrorCode::OutOfRange, "out of memory for levels"));
    }
    u64 offset = 0;
    u32 width = texture.width;
    u32 height = texture.height;
    for (u32 level = 0; level < texture.mipCount; ++level)
    {
        u32 rowPitch = 0;
        auto size = TextureLevelSize(texture.format, width, height, rowPitch);
        if (size.IsError())
        {
            return ImportStatus::Err(size.Error());
        }
        TextureLevel& record = texture.levels[level];
        record.offset = offset;
        record.size = size.Value() * texture.layerCount;
        record.width = width;
        record.height = height;
        record.rowPitch = rowPitch;
        offset += record.size;
        width = width > 1 ? width / 2 : 1;
        height = height > 1 ? height / 2 : 1;
    }
    if (offset > static_cast<u64>(static_cast<usize>(-1)) ||
        !texture.data.TryResize(static_cast<usize>(offset)))
    {
        return ImportStatus::Err(MakeError(ErrorCode::OutOfRange, "out of memory for texels"));
    }
    return ImportStatus::Ok();
}

ImportResult<TextureAsset> MakeShape(const TextureAsset& source, TextureFormat format, u32 mipCount,
                                     Allocator& allocator) noexcept
{
    TextureAsset texture(allocator);
    texture.format = format;
    texture.width = source.width;
    texture.height = source.height;
    texture.depth = 1;
    texture.mipCount = mipCount;
    texture.layerCount = source.layerCount;
    texture.cubemap = source.cubemap;
    if (!texture.name.TryAssign(StringView(source.name.Data(), source.name.Size())))
    {
        return ImportResult<TextureAsset>::Err(
            MakeError(ErrorCode::OutOfRange, "out of memory for a texture name"));
    }
    auto status = BuildLevelTable(texture);
    if (status.IsError())
    {
        return ImportResult<TextureAsset>::Err(status.Error());
    }
    return ImportResult<TextureAsset>::Ok(std::move(texture));
}

}  // namespace

ImportResult<TextureAsset> CompressTexture(const TextureAsset& source, TextureFormat target,
                                           const BcEncodeSettings& settings,
                                           Allocator& allocator) noexcept
{
    if (source.format != TextureFormat::Rgba8Unorm)
    {
        return ImportResult<TextureAsset>::Err(
            MakeError(ErrorCode::InvalidArgument, "compression source must be rgba8"));
    }
    if (!IsBcCodecFormat(target))
    {
        return ImportResult<TextureAsset>::Err(
            MakeError(ErrorCode::Unsupported, "target format has no encoder"));
    }
    if (source.depth != 1)
    {
        return ImportResult<TextureAsset>::Err(
            MakeError(ErrorCode::Unsupported, "volume textures cannot be block compressed"));
    }
    auto valid = source.Validate();
    if (valid.IsError())
    {
        return ImportResult<TextureAsset>::Err(valid.Error());
    }
    auto shaped = MakeShape(source, target, source.mipCount, allocator);
    if (shaped.IsError())
    {
        return shaped;
    }
    TextureAsset texture = std::move(shaped.Value());
    const u32 blockBytes = TextureBlockBytes(target);

    for (u32 level = 0; level < source.mipCount; ++level)
    {
        const TextureLevel& in = source.levels[level];
        const TextureLevel& out = texture.levels[level];
        const u64 surfaceIn = in.size / source.layerCount;
        const u64 surfaceOut = out.size / texture.layerCount;
        const u32 blocksX = TextureBlocksAcross(in.width);
        const u32 blocksY = TextureBlocksAcross(in.height);
        for (u32 layer = 0; layer < source.layerCount; ++layer)
        {
            const Byte* pixels = source.data.Data() + in.offset + surfaceIn * layer;
            Byte* blocks = texture.data.Data() + out.offset + surfaceOut * layer;
            for (u32 by = 0; by < blocksY; ++by)
            {
                for (u32 bx = 0; bx < blocksX; ++bx)
                {
                    u8 rgba[kBcRgbaBlockBytes];
                    for (u32 y = 0; y < 4; ++y)
                    {
                        const u32 sy = std::min(by * 4 + y, in.height - 1);
                        for (u32 x = 0; x < 4; ++x)
                        {
                            const u32 sx = std::min(bx * 4 + x, in.width - 1);
                            std::memcpy(rgba + (y * 4 + x) * 4,
                                        pixels + static_cast<u64>(sy) * in.rowPitch + sx * 4u, 4);
                        }
                    }
                    Byte* destination = blocks + (static_cast<u64>(by) * blocksX + bx) * blockBytes;
                    switch (target)
                    {
                    case TextureFormat::Bc1Rgba:
                        EncodeBc1Block(rgba, destination, settings);
                        break;
                    case TextureFormat::Bc4R:
                    {
                        u8 red[16];
                        for (u32 t = 0; t < 16; ++t)
                        {
                            red[t] = rgba[t * 4];
                        }
                        EncodeBc4Block(red, destination, settings);
                        break;
                    }
                    case TextureFormat::Bc5Rg:
                    {
                        u8 redGreen[32];
                        for (u32 t = 0; t < 16; ++t)
                        {
                            redGreen[t * 2] = rgba[t * 4];
                            redGreen[t * 2 + 1] = rgba[t * 4 + 1];
                        }
                        EncodeBc5Block(redGreen, destination, settings);
                        break;
                    }
                    case TextureFormat::Bc2Rgba:
                        EncodeBc2Block(rgba, destination, settings);
                        break;
                    case TextureFormat::Bc3Rgba:
                        EncodeBc3Block(rgba, destination, settings);
                        break;
                    default:
                        EncodeBc7Block(rgba, destination, settings);
                        break;
                    }
                }
            }
        }
    }
    return ImportResult<TextureAsset>::Ok(std::move(texture));
}

ImportResult<TextureAsset> DecompressTexture(const TextureAsset& source,
                                             Allocator& allocator) noexcept
{
    if (!IsBcCodecFormat(source.format))
    {
        return ImportResult<TextureAsset>::Err(
            MakeError(ErrorCode::Unsupported, "texture format has no decoder"));
    }
    auto valid = source.Validate();
    if (valid.IsError())
    {
        return ImportResult<TextureAsset>::Err(valid.Error());
    }
    auto shaped = MakeShape(source, TextureFormat::Rgba8Unorm, source.mipCount, allocator);
    if (shaped.IsError())
    {
        return shaped;
    }
    TextureAsset texture = std::move(shaped.Value());
    const u32 blockBytes = TextureBlockBytes(source.format);

    for (u32 level = 0; level < source.mipCount; ++level)
    {
        const TextureLevel& in = source.levels[level];
        const TextureLevel& out = texture.levels[level];
        const u64 surfaceIn = in.size / source.layerCount;
        const u64 surfaceOut = out.size / texture.layerCount;
        const u32 blocksX = TextureBlocksAcross(in.width);
        const u32 blocksY = TextureBlocksAcross(in.height);
        for (u32 layer = 0; layer < source.layerCount; ++layer)
        {
            const Byte* blocks = source.data.Data() + in.offset + surfaceIn * layer;
            Byte* pixels = texture.data.Data() + out.offset + surfaceOut * layer;
            for (u32 by = 0; by < blocksY; ++by)
            {
                for (u32 bx = 0; bx < blocksX; ++bx)
                {
                    const Byte* block = blocks + (static_cast<u64>(by) * blocksX + bx) * blockBytes;
                    u8 rgba[kBcRgbaBlockBytes];
                    switch (source.format)
                    {
                    case TextureFormat::Bc1Rgba:
                        DecodeBc1Block(block, rgba);
                        break;
                    case TextureFormat::Bc2Rgba:
                        DecodeBc2Block(block, rgba);
                        break;
                    case TextureFormat::Bc3Rgba:
                        DecodeBc3Block(block, rgba);
                        break;
                    case TextureFormat::Bc4R:
                    {
                        u8 red[16];
                        DecodeBc4Block(block, red);
                        for (u32 t = 0; t < 16; ++t)
                        {
                            rgba[t * 4] = red[t];
                            rgba[t * 4 + 1] = 0;
                            rgba[t * 4 + 2] = 0;
                            rgba[t * 4 + 3] = 255;
                        }
                        break;
                    }
                    case TextureFormat::Bc5Rg:
                    {
                        u8 redGreen[32];
                        DecodeBc5Block(block, redGreen);
                        for (u32 t = 0; t < 16; ++t)
                        {
                            rgba[t * 4] = redGreen[t * 2];
                            rgba[t * 4 + 1] = redGreen[t * 2 + 1];
                            rgba[t * 4 + 2] = 0;
                            rgba[t * 4 + 3] = 255;
                        }
                        break;
                    }
                    default:
                        DecodeBc7Block(block, rgba);
                        break;
                    }
                    for (u32 y = 0; y < 4; ++y)
                    {
                        const u32 dy = by * 4 + y;
                        if (dy >= in.height)
                        {
                            break;
                        }
                        for (u32 x = 0; x < 4; ++x)
                        {
                            const u32 dx = bx * 4 + x;
                            if (dx >= in.width)
                            {
                                break;
                            }
                            std::memcpy(pixels + static_cast<u64>(dy) * out.rowPitch + dx * 4u,
                                        rgba + (y * 4 + x) * 4, 4);
                        }
                    }
                }
            }
        }
    }
    return ImportResult<TextureAsset>::Ok(std::move(texture));
}

namespace
{

struct SrgbDecodeTable
{
    f32 linear[256] = {};
    SrgbDecodeTable() noexcept
    {
        for (u32 value = 0; value < 256; ++value)
        {
            const f32 c = static_cast<f32>(value) / 255.0f;
            linear[value] = c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
        }
    }
};

u8 EncodeSrgb(f32 linear) noexcept
{
    const f32 c = linear <= 0.0031308f
                      ? linear * 12.92f
                      : 1.055f * std::pow(std::max(linear, 0.0f), 1.0f / 2.4f) - 0.055f;
    return static_cast<u8>(std::clamp(c * 255.0f + 0.5f, 0.0f, 255.0f));
}

}  // namespace

ImportResult<TextureAsset> GenerateMipChain(const TextureAsset& source, bool srgb,
                                            Allocator& allocator) noexcept
{
    if (source.format != TextureFormat::Rgba8Unorm || source.depth != 1)
    {
        return ImportResult<TextureAsset>::Err(
            MakeError(ErrorCode::InvalidArgument, "mip generation needs a 2D rgba8 texture"));
    }
    auto valid = source.Validate();
    if (valid.IsError())
    {
        return ImportResult<TextureAsset>::Err(valid.Error());
    }
    u32 mipCount = 1;
    for (u32 size = std::max(source.width, source.height); size > 1; size /= 2)
    {
        ++mipCount;
    }
    auto shaped = MakeShape(source, TextureFormat::Rgba8Unorm, mipCount, allocator);
    if (shaped.IsError())
    {
        return shaped;
    }
    TextureAsset texture = std::move(shaped.Value());
    static const SrgbDecodeTable decode{};

    for (u32 layer = 0; layer < source.layerCount; ++layer)
    {
        const TextureLevel& base = source.levels[0];
        const u64 baseSurface = base.size / source.layerCount;
        std::memcpy(texture.data.Data() + texture.levels[0].offset + baseSurface * layer,
                    source.data.Data() + base.offset + baseSurface * layer,
                    static_cast<usize>(baseSurface));
        for (u32 level = 1; level < mipCount; ++level)
        {
            const TextureLevel& parent = texture.levels[level - 1];
            const TextureLevel& child = texture.levels[level];
            const Byte* in =
                texture.data.Data() + parent.offset + (parent.size / source.layerCount) * layer;
            Byte* out =
                texture.data.Data() + child.offset + (child.size / source.layerCount) * layer;
            for (u32 y = 0; y < child.height; ++y)
            {
                // Odd parents fold their last row/column into the final child
                // texel so no source texel is dropped.
                const u32 y0 = std::min(y * 2, parent.height - 1);
                u32 rows = parent.height == 1 ? 1u : 2u;
                if ((parent.height & 1u) != 0 && y == child.height - 1 && parent.height > 1)
                {
                    rows = 3;
                }
                for (u32 x = 0; x < child.width; ++x)
                {
                    const u32 x0 = std::min(x * 2, parent.width - 1);
                    u32 columns = parent.width == 1 ? 1u : 2u;
                    if ((parent.width & 1u) != 0 && x == child.width - 1 && parent.width > 1)
                    {
                        columns = 3;
                    }
                    f32 sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (u32 dy = 0; dy < rows; ++dy)
                    {
                        const Byte* row = in + static_cast<u64>(y0 + dy) * parent.rowPitch;
                        for (u32 dx = 0; dx < columns; ++dx)
                        {
                            const Byte* texel = row + (x0 + dx) * 4u;
                            for (u32 ch = 0; ch < 4; ++ch)
                            {
                                const u8 value = static_cast<u8>(texel[ch]);
                                sum[ch] += (srgb && ch < 3) ? decode.linear[value]
                                                            : static_cast<f32>(value);
                            }
                        }
                    }
                    const f32 inverse = 1.0f / static_cast<f32>(rows * columns);
                    Byte* destination = out + static_cast<u64>(y) * child.rowPitch + x * 4u;
                    for (u32 ch = 0; ch < 4; ++ch)
                    {
                        const f32 mean = sum[ch] * inverse;
                        destination[ch] = static_cast<Byte>(
                            (srgb && ch < 3)
                                ? EncodeSrgb(mean)
                                : static_cast<u8>(std::clamp(mean + 0.5f, 0.0f, 255.0f)));
                    }
                }
            }
        }
    }
    return ImportResult<TextureAsset>::Ok(std::move(texture));
}

f64 ComputePsnr(Span<const u8> lhs, Span<const u8> rhs, u32 width, u32 height,
                u32 channelMask) noexcept
{
    const u64 texels = static_cast<u64>(width) * height;
    if (texels == 0 || (channelMask & 0xFu) == 0 || lhs.Size() < texels * 4 ||
        rhs.Size() < texels * 4)
    {
        return 0.0;
    }
    u64 squared = 0;
    u64 samples = 0;
    for (u64 texel = 0; texel < texels; ++texel)
    {
        for (u32 ch = 0; ch < 4; ++ch)
        {
            if ((channelMask & (1u << ch)) == 0)
            {
                continue;
            }
            const i32 diff =
                static_cast<i32>(lhs[texel * 4 + ch]) - static_cast<i32>(rhs[texel * 4 + ch]);
            squared += static_cast<u64>(diff * diff);
            ++samples;
        }
    }
    if (squared == 0)
    {
        return std::numeric_limits<f64>::infinity();
    }
    const f64 mse = static_cast<f64>(squared) / static_cast<f64>(samples);
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

}  // namespace cyber::assets
