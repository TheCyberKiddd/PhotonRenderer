// BC6H (HDR) block codec: half-float conversion, hand-built decoder vectors,
// encoder quality floors on an HDR test card, the signed variant, and edge
// cases. The decoder is also compared with the hardware decoder of every local
// adapter in the renderer's `renderer_bc6h_hardware` test.

#include <cyber/assets/bc_codec.h>
#include <cyber/test/test.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{

using cyber::assets::BcEncodeSettings;
using cyber::assets::BcQuality;
using cyber::assets::DecodeBc6hBlock;
using cyber::assets::EncodeBc6hBlock;
using cyber::assets::FloatToHalf;
using cyber::assets::HalfToFloat;
using cyber::core::Byte;
using cyber::core::f32;
using cyber::core::f64;
using cyber::core::u16;
using cyber::core::u32;
using cyber::memory::Span;

// Encoder quality floors in dB on the tone-mapped HDR card below, about half a
// decibel under what the encoder measures. An encoder change that lowers
// quality under them fails CI; raise them when quality improves.
constexpr f64 kMinPsnrFast = 34.5;
constexpr f64 kMinPsnrNormal = 38.0;
constexpr f64 kMinPsnrHigh = 38.0;
constexpr f64 kMinPsnrSigned = 37.8;

constexpr u32 kSize = 64;

u32 NextRandom(u32& state)
{
    state = state * 1664525u + 1013904223u;
    return state >> 8u;
}

// An HDR test card in half floats, four values per texel: an exponential
// gradient from 0.001 to ~60, hard stripes between 0.02 and 200, a colour ring
// with a wide brightness ramp, and multiplicative noise.
std::vector<u16> MakeHdrCard(bool withNegatives)
{
    std::vector<u16> texels(static_cast<size_t>(kSize) * kSize * 4u);
    u32 seed = 987654u;
    for (u32 y = 0; y < kSize; ++y)
    {
        for (u32 x = 0; x < kSize; ++x)
        {
            f32 rgb[3] = {0, 0, 0};
            if (y < kSize / 2 && x < kSize / 2)
            {
                rgb[0] = 0.001f * std::exp(static_cast<f32>(x) * 0.19f);
                rgb[1] = 0.002f * std::exp(static_cast<f32>(y) * 0.17f);
                rgb[2] = 0.5f + 0.3f * std::sin(static_cast<f32>(x + y) * 0.2f);
            }
            else if (y < kSize / 2)
            {
                const bool stripe = ((x / 3 + y / 5) & 1u) != 0;
                rgb[0] = stripe ? 200.0f : 0.02f;
                rgb[1] = stripe ? 0.5f : 30.0f;
                rgb[2] = stripe ? 3.0f : 0.1f;
            }
            else if (x < kSize / 2)
            {
                const f32 angle = std::atan2(static_cast<f32>(y) - kSize * 0.75f,
                                             static_cast<f32>(x) - kSize * 0.25f);
                const f32 gain = std::exp((static_cast<f32>(x) - 16.0f) * 0.3f);
                rgb[0] = gain * (1.0f + std::cos(angle));
                rgb[1] = gain * (1.0f + std::cos(angle - 2.094f));
                rgb[2] = gain * (1.0f + std::cos(angle + 2.094f));
            }
            else
            {
                const f32 base = 0.05f * std::exp(static_cast<f32>(x - 32) * 0.12f);
                for (u32 c = 0; c < 3; ++c)
                {
                    rgb[c] =
                        base * (0.7f + 0.6f * static_cast<f32>(NextRandom(seed) % 1000u) / 1000.0f);
                }
            }
            if (withNegatives && ((x / 8 + y / 8) & 1u) != 0)
            {
                rgb[0] = -rgb[0];
                rgb[2] = -rgb[2];
            }
            u16* texel = texels.data() + (static_cast<size_t>(y) * kSize + x) * 4u;
            for (u32 c = 0; c < 3; ++c)
            {
                texel[c] = FloatToHalf(rgb[c]);
            }
            texel[3] = 0x3C00;
        }
    }
    return texels;
}

// PSNR of the tone-mapped images (x / (1 + |x|) per channel, sign kept), so a
// dark detail and a bright highlight count alike.
f64 ToneMappedPsnr(const std::vector<u16>& reference, const std::vector<u16>& decoded)
{
    f64 error = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < reference.size(); i += 4)
    {
        for (u32 c = 0; c < 3; ++c)
        {
            const f64 a = HalfToFloat(reference[i + c]);
            const f64 b = HalfToFloat(decoded[i + c]);
            const f64 ta = a / (1.0 + std::fabs(a));
            const f64 tb = b / (1.0 + std::fabs(b));
            error += (ta - tb) * (ta - tb);
            ++count;
        }
    }
    const f64 mse = error / static_cast<f64>(count);
    return mse <= 0.0 ? 200.0 : -10.0 * std::log10(mse);
}

struct BitPacker
{
    Byte bytes[16] = {};
    u32 position = 0;
    void Put(u32 value, u32 count)
    {
        for (u32 bit = 0; bit < count; ++bit, ++position)
        {
            if ((value >> bit) & 1u)
            {
                bytes[position >> 3u] = static_cast<Byte>(static_cast<u32>(bytes[position >> 3u]) |
                                                          (1u << (position & 7u)));
            }
        }
    }
};

f64 RoundTrip(const std::vector<u16>& card, BcQuality quality, bool isSigned)
{
    BcEncodeSettings settings;
    settings.quality = quality;
    std::vector<Byte> blocks(cyber::assets::Bc6hSurfaceBytes(kSize, kSize));
    if (!cyber::assets::CompressBc6hSurface(Span<const u16>(card.data(), card.size()), kSize, kSize,
                                            Span<Byte>(blocks.data(), blocks.size()), settings,
                                            isSigned))
    {
        return 0.0;
    }
    std::vector<u16> decoded(card.size());
    if (!cyber::assets::DecompressBc6hSurface(Span<const Byte>(blocks.data(), blocks.size()), kSize,
                                              kSize, Span<u16>(decoded.data(), decoded.size()),
                                              isSigned))
    {
        return 0.0;
    }
    return ToneMappedPsnr(card, decoded);
}

}  // namespace

CG_TEST(half_float_conversion_is_exact_and_rounds_to_nearest_even)
{
    u32 failures = 0;
    for (u32 pattern = 0; pattern < 65536; ++pattern)
    {
        const bool nan = ((pattern >> 10u) & 0x1Fu) == 31u && (pattern & 0x3FFu) != 0;
        if (nan)
        {
            CG_EXPECT_TRUE(std::isnan(HalfToFloat(static_cast<u16>(pattern))));
            continue;
        }
        failures += FloatToHalf(HalfToFloat(static_cast<u16>(pattern))) != pattern ? 1u : 0u;
    }
    CG_EXPECT_EQ(failures, 0u);
    CG_EXPECT_EQ(static_cast<u32>(FloatToHalf(1.0f)), 0x3C00u);
    CG_EXPECT_EQ(static_cast<u32>(FloatToHalf(-2.0f)), 0xC000u);
    CG_EXPECT_EQ(static_cast<u32>(FloatToHalf(65504.0f)), 0x7BFFu);
    CG_EXPECT_EQ(static_cast<u32>(FloatToHalf(65520.0f)), 0x7C00u);  // halfway: to infinity
    CG_EXPECT_EQ(static_cast<u32>(FloatToHalf(65519.0f)), 0x7BFFu);
    CG_EXPECT_EQ(static_cast<u32>(FloatToHalf(std::ldexp(1.0f, -24))),
                 0x0001u);  // smallest subnormal
    CG_EXPECT_EQ(static_cast<u32>(FloatToHalf(std::ldexp(1.0f, -25))), 0x0000u);  // tie to even
    CG_EXPECT_EQ(static_cast<u32>(FloatToHalf(std::ldexp(1.5f, -25))), 0x0001u);
    CG_EXPECT_EQ(static_cast<u32>(FloatToHalf(1.0f + std::ldexp(1.0f, -11))), 0x3C00u);
    CG_EXPECT_EQ(static_cast<u32>(FloatToHalf(1.0f + std::ldexp(3.0f, -11))), 0x3C02u);
    CG_EXPECT_EQ(HalfToFloat(0x3C00), 1.0f);
    CG_EXPECT_EQ(HalfToFloat(0x0001), std::ldexp(1.0f, -24));
}

CG_TEST(bc6h_decoder_matches_hand_built_blocks_and_zeroes_reserved_modes)
{
    // An all-zero block is mode 1 with every endpoint and index zero: black.
    Byte zero[16] = {};
    u16 texels[48];
    std::memset(texels, 0xFF, sizeof(texels));
    DecodeBc6hBlock(zero, texels, false);
    for (const u16 value : texels)
    {
        CG_EXPECT_EQ(static_cast<u32>(value), 0u);
    }

    // Mode 11: endpoint 0 = (0, 0, 0), endpoint 1 = (1023, 1023, 1023) at 10 bits,
    // texel i has index i (the anchor, texel 0, stores one bit fewer).
    BitPacker packer;
    packer.Put(0x03, 5);
    packer.Put(0, 10);
    packer.Put(0, 10);
    packer.Put(0, 10);
    packer.Put(1023, 10);
    packer.Put(1023, 10);
    packer.Put(1023, 10);
    for (u32 texel = 0; texel < 16; ++texel)
    {
        packer.Put(texel, texel == 0 ? 3 : 4);
    }
    CG_EXPECT_EQ(packer.position, 128u);
    constexpr u32 kWeights4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};
    DecodeBc6hBlock(packer.bytes, texels, false);
    for (u32 texel = 0; texel < 16; ++texel)
    {
        const u32 mixed = (65535u * kWeights4[texel] + 32u) >> 6u;
        const u32 expected = (mixed * 31u) >> 6u;
        for (u32 c = 0; c < 3; ++c)
        {
            CG_EXPECT_EQ(static_cast<u32>(texels[texel * 3 + c]), expected);
        }
    }
    CG_EXPECT_EQ(static_cast<u32>(texels[15 * 3]), 0x7BFFu);  // the largest finite half

    // The reserved mode values decode to black.
    for (const u32 reserved : {0x13u, 0x17u, 0x1Bu, 0x1Fu})
    {
        BitPacker block;
        block.Put(reserved, 5);
        block.Put(0xFFFFFFFFu, 32);
        block.Put(0xFFFFFFFFu, 32);
        std::memset(texels, 0xFF, sizeof(texels));
        DecodeBc6hBlock(block.bytes, texels, false);
        for (const u16 value : texels)
        {
            CG_EXPECT_EQ(static_cast<u32>(value), 0u);
        }
    }

    // The signed decoder reads endpoint 1 = -1 at 10 bits as a small negative.
    BitPacker negative;
    negative.Put(0x03, 5);
    negative.Put(0, 30);
    negative.Put(1023, 10);
    negative.Put(1023, 10);
    negative.Put(1023, 10);
    for (u32 texel = 0; texel < 16; ++texel)
    {
        negative.Put(texel == 0 ? 3u : 15u, texel == 0 ? 3 : 4);
    }
    DecodeBc6hBlock(negative.bytes, texels, true);
    CG_EXPECT_TRUE((texels[15 * 3] & 0x8000u) != 0);
    CG_EXPECT_TRUE((texels[0] & 0x8000u) != 0);
}

CG_TEST(bc6h_encoder_round_trips_with_floors_and_more_search_never_hurts)
{
    const std::vector<u16> card = MakeHdrCard(false);
    const f64 fast = RoundTrip(card, BcQuality::Fast, false);
    const f64 normal = RoundTrip(card, BcQuality::Normal, false);
    const f64 high = RoundTrip(card, BcQuality::High, false);
    std::printf("bc6h tone-mapped PSNR: fast %.2f, normal %.2f, high %.2f dB\n", fast, normal,
                high);
    CG_EXPECT_TRUE(normal >= fast - 0.01);
    CG_EXPECT_TRUE(high >= normal - 0.01);
    CG_EXPECT_TRUE(fast >= kMinPsnrFast);
    CG_EXPECT_TRUE(normal >= kMinPsnrNormal);
    CG_EXPECT_TRUE(high >= kMinPsnrHigh);
}

CG_TEST(bc6h_signed_variant_round_trips)
{
    const std::vector<u16> card = MakeHdrCard(true);
    const f64 psnr = RoundTrip(card, BcQuality::Normal, true);
    std::printf("bc6h signed tone-mapped PSNR: %.2f dB\n", psnr);
    CG_EXPECT_TRUE(psnr >= kMinPsnrSigned);
}

CG_TEST(bc6h_solid_blocks_and_extremes_stay_close_and_encoding_is_deterministic)
{
    BcEncodeSettings settings;
    settings.quality = BcQuality::Normal;
    // Solid colours across the range, including zero and the largest finite half.
    const f32 levels[] = {0.0f, 1.0e-4f, 0.25f, 1.0f, 17.5f, 4096.0f, 65504.0f};
    for (const f32 level : levels)
    {
        u16 texels[48];
        for (u32 i = 0; i < 48; i += 3)
        {
            texels[i] = FloatToHalf(level);
            texels[i + 1] = FloatToHalf(level * 0.5f);
            texels[i + 2] = FloatToHalf(level * 0.25f);
        }
        Byte first[16];
        Byte second[16];
        EncodeBc6hBlock(texels, first, settings, false);
        EncodeBc6hBlock(texels, second, settings, false);
        CG_EXPECT_TRUE(std::memcmp(first, second, 16) == 0);
        u16 decoded[48];
        DecodeBc6hBlock(first, decoded, false);
        for (u32 i = 0; i < 48; ++i)
        {
            const f32 want = HalfToFloat(texels[i]);
            const f32 got = HalfToFloat(decoded[i]);
            // Within a quantization step of the 10 to 16 bit endpoints: well
            // under one percent of the value, or a tiny amount near zero.
            CG_EXPECT_TRUE(std::fabs(got - want) <= 0.004f * want + 2.0e-4f);
        }
    }
    // Infinity and NaN clamp to the largest finite half instead of poisoning the
    // block, and negatives clamp to zero when unsigned.
    u16 odd[48];
    for (u32 i = 0; i < 48; ++i)
    {
        odd[i] = (i % 3 == 0) ? 0x7C00 : ((i % 3 == 1) ? 0xBC00 : 0x7E00);
    }
    Byte block[16];
    EncodeBc6hBlock(odd, block, settings, false);
    u16 decoded[48];
    DecodeBc6hBlock(block, decoded, false);
    for (u32 i = 0; i < 48; ++i)
    {
        const f32 value = HalfToFloat(decoded[i]);
        CG_EXPECT_TRUE(std::isfinite(value));
        CG_EXPECT_TRUE(value >= 0.0f);
    }
}
