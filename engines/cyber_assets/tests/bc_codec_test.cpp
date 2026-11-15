#include <cyber/assets/bc_codec.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/test/test.h>

#include <cmath>
#include <cstdio>
#include <cstring>

namespace
{

using cyber::assets::BcEncodeSettings;
using cyber::assets::BcQuality;
using cyber::assets::CompressTexture;
using cyber::assets::ComputePsnr;
using cyber::assets::DecompressTexture;
using cyber::assets::GenerateMipChain;
using cyber::assets::TextureAsset;
using cyber::assets::TextureFormat;
using cyber::core::Byte;
using cyber::core::f64;
using cyber::core::u32;
using cyber::core::u8;
using cyber::memory::HeapAllocator;
using cyber::memory::Span;

// Encoder quality floors in dB, measured against the decoder on the synthetic
// image below with a margin of about half a decibel. An encoder change that
// lowers quality below these fails CI; raise them when quality improves.
constexpr f64 kMinPsnrBc1 = 37.0;
constexpr f64 kMinPsnrBc4 = 51.0;
constexpr f64 kMinPsnrBc5 = 51.0;
constexpr f64 kMinPsnrBc7Fast = 43.0;
constexpr f64 kMinPsnrBc7Normal = 43.0;
constexpr f64 kMinPsnrBc7Alpha = 39.5;
constexpr f64 kMinPsnrBc2 = 36.0;
constexpr f64 kMinPsnrBc3 = 38.0;
constexpr f64 kMinPsnrBc7AlphaNormal = 40.8;
constexpr f64 kMinPsnrBc7AlphaHigh = 42.0;

constexpr u32 kSize = 128;

u32 NextRandom(u32& state)
{
    state = state * 1664525u + 1013904223u;
    return state >> 8u;
}

// A deterministic test card: smooth gradients (where endpoint precision
// shows), hard two-colour edges (where partitions matter), a hue ring (where
// colour lines bend inside a block), and low-amplitude noise (texture detail).
TextureAsset MakeTestCard(HeapAllocator& allocator, bool withAlpha)
{
    TextureAsset texture(allocator);
    texture.format = TextureFormat::Rgba8Unorm;
    texture.width = kSize;
    texture.height = kSize;
    texture.mipCount = 1;
    (void)texture.data.TryResize(kSize * kSize * 4);
    (void)texture.levels.TryResize(1);
    texture.levels[0].width = kSize;
    texture.levels[0].height = kSize;
    texture.levels[0].rowPitch = kSize * 4;
    texture.levels[0].size = kSize * kSize * 4;

    u32 seed = 12345u;
    for (u32 y = 0; y < kSize; ++y)
    {
        for (u32 x = 0; x < kSize; ++x)
        {
            f64 r = 0.0;
            f64 g = 0.0;
            f64 b = 0.0;
            if (y < kSize / 2 && x < kSize / 2)
            {
                r = x * 4.0;
                g = y * 4.0;
                b = 128.0 + 60.0 * std::sin((x + y) * 0.1);
            }
            else if (y < kSize / 2)
            {
                const bool stripe = ((x / 3 + y / 5) & 1u) != 0;
                r = stripe ? 230.0 : 30.0;
                g = stripe ? 40.0 : 200.0;
                b = stripe ? 60.0 : 90.0;
            }
            else if (x < kSize / 2)
            {
                const f64 angle = std::atan2(y - kSize * 0.75, x - kSize * 0.25);
                r = 127.5 + 127.5 * std::cos(angle);
                g = 127.5 + 127.5 * std::cos(angle - 2.094);
                b = 127.5 + 127.5 * std::cos(angle + 2.094);
            }
            else
            {
                const f64 base = 90.0 + 0.5 * x;
                r = base + static_cast<f64>(NextRandom(seed) % 24u);
                g = base * 0.8 + static_cast<f64>(NextRandom(seed) % 24u);
                b = base * 0.6 + static_cast<f64>(NextRandom(seed) % 24u);
            }
            Byte* texel = texture.data.Data() + (y * kSize + x) * 4u;
            texel[0] = static_cast<Byte>(std::fmin(std::fmax(r, 0.0), 255.0));
            texel[1] = static_cast<Byte>(std::fmin(std::fmax(g, 0.0), 255.0));
            texel[2] = static_cast<Byte>(std::fmin(std::fmax(b, 0.0), 255.0));
            texel[3] = static_cast<Byte>(withAlpha ? (x * 2u + y) & 0xFFu : 255u);
        }
    }
    return texture;
}

Span<const u8> Pixels(const TextureAsset& texture)
{
    return Span<const u8>(reinterpret_cast<const u8*>(texture.data.Data()), texture.data.Size());
}

f64 RoundTripPsnr(const TextureAsset& source, TextureFormat format, BcQuality quality,
                  u32 channelMask, HeapAllocator& allocator)
{
    BcEncodeSettings settings;
    settings.quality = quality;
    settings.bc1AlphaThreshold = 0;
    auto compressed = CompressTexture(source, format, settings, allocator);
    if (compressed.IsError())
    {
        return 0.0;
    }
    auto decoded = DecompressTexture(compressed.Value(), allocator);
    if (decoded.IsError())
    {
        return 0.0;
    }
    const f64 psnr = ComputePsnr(Pixels(source), Pixels(decoded.Value()), source.width,
                                 source.height, channelMask);
    std::printf("  %s quality %u: %.2f dB\n", cyber::assets::TextureFormatName(format),
                static_cast<u32>(quality), psnr);
    return psnr;
}

}  // namespace

CG_TEST(bc1_decoder_matches_the_specification_palette)
{
    // c0 = pure red (0xF800) > c1 = pure blue (0x001F): four-colour mode.
    // Indices 0,1,2,3 repeat across the block.
    const Byte block[8] = {Byte{0x00}, Byte{0xF8}, Byte{0x1F}, Byte{0x00},
                           Byte{0xE4}, Byte{0xE4}, Byte{0xE4}, Byte{0xE4}};
    u8 rgba[64];
    cyber::assets::DecodeBc1Block(block, rgba);
    CG_EXPECT_EQ(rgba[0], 255u);
    CG_EXPECT_EQ(rgba[2], 0u);
    CG_EXPECT_EQ(rgba[4 + 0], 0u);
    CG_EXPECT_EQ(rgba[4 + 2], 255u);
    CG_EXPECT_EQ(rgba[8 + 0], 170u);  // (2 * 255 + 0 + 1) / 3
    CG_EXPECT_EQ(rgba[8 + 2], 85u);
    CG_EXPECT_EQ(rgba[12 + 0], 85u);
    CG_EXPECT_EQ(rgba[12 + 3], 255u);

    // Swapped endpoints select three-colour mode: index 3 is transparent black.
    const Byte three[8] = {Byte{0x1F}, Byte{0x00}, Byte{0x00}, Byte{0xF8},
                           Byte{0xE4}, Byte{0xE4}, Byte{0xE4}, Byte{0xE4}};
    cyber::assets::DecodeBc1Block(three, rgba);
    CG_EXPECT_EQ(rgba[8 + 0], 128u);  // (0 + 255 + 1) / 2
    CG_EXPECT_EQ(rgba[8 + 2], 128u);
    CG_EXPECT_EQ(rgba[12 + 3], 0u);
}

CG_TEST(bc4_decoder_covers_both_modes)
{
    // a = 255 > b = 0: eight values. Texel 0 code 2, texel 1 code 7.
    const Byte eight[8] = {Byte{255}, Byte{0}, Byte{0x3A}, Byte{0},
                           Byte{0},   Byte{0}, Byte{0},    Byte{0}};
    u8 red[16];
    cyber::assets::DecodeBc4Block(eight, red);
    CG_EXPECT_EQ(red[0], 219u);  // (6 * 255 + 3) / 7
    CG_EXPECT_EQ(red[1], 36u);   // (1 * 255 + 3) / 7
    // a = 10 <= b = 20: six values plus 0 and 255. Codes 6, 7, 2.
    const Byte six[8] = {Byte{10}, Byte{20}, Byte{0xBE}, Byte{0},
                         Byte{0},  Byte{0},  Byte{0},    Byte{0}};
    cyber::assets::DecodeBc4Block(six, red);
    CG_EXPECT_EQ(red[0], 0u);
    CG_EXPECT_EQ(red[1], 255u);
    CG_EXPECT_EQ(red[2], 12u);  // (4 * 10 + 1 * 20 + 2) / 5
}

CG_TEST(bc7_decoder_handles_reserved_and_mode_6_blocks)
{
    Byte block[16] = {};
    u8 rgba[64];
    std::memset(rgba, 0xAB, sizeof(rgba));
    cyber::assets::DecodeBc7Block(block, rgba);
    for (const u8 value : rgba)
    {
        CG_EXPECT_EQ(value, 0u);
    }

    // Mode 6 with every endpoint code 127 and every p-bit 1 decodes to opaque
    // white whatever the indices are: (127 << 1 | 1) = 255.
    block[0] = Byte{0x40};
    // Endpoint fields: 8 x 7 bits from bit 7, all ones, then the two p-bits.
    for (u32 bit = 7; bit < 7 + 56 + 2; ++bit)
    {
        block[bit / 8] = static_cast<Byte>(static_cast<u32>(block[bit / 8]) | (1u << (bit % 8)));
    }
    cyber::assets::DecodeBc7Block(block, rgba);
    for (const u8 value : rgba)
    {
        CG_EXPECT_EQ(value, 255u);
    }
}

CG_TEST(solid_blocks_round_trip_exactly)
{
    const BcEncodeSettings settings;
    const u8 colors[][4] = {{0, 0, 0, 255},
                            {255, 255, 255, 255},
                            {37, 141, 222, 255},
                            {128, 128, 128, 255},
                            {201, 13, 99, 255}};
    for (const auto& color : colors)
    {
        u8 rgba[64];
        for (u32 texel = 0; texel < 16; ++texel)
        {
            std::memcpy(rgba + texel * 4, color, 4);
        }
        Byte block[16];
        u8 decoded[64];
        cyber::assets::EncodeBc7Block(rgba, block, settings);
        cyber::assets::DecodeBc7Block(block, decoded);
        CG_EXPECT_EQ(std::memcmp(rgba, decoded, 64), 0);

        u8 red[16];
        std::memset(red, color[0], sizeof(red));
        u8 redOut[16];
        cyber::assets::EncodeBc4Block(red, block, settings);
        cyber::assets::DecodeBc4Block(block, redOut);
        CG_EXPECT_EQ(std::memcmp(red, redOut, 16), 0);

        // BC1 cannot store every colour exactly; the single-colour tables
        // keep it within the 565 interpolation grid.
        cyber::assets::EncodeBc1Block(rgba, block, settings);
        cyber::assets::DecodeBc1Block(block, decoded);
        for (u32 ch = 0; ch < 3; ++ch)
        {
            CG_EXPECT_TRUE(std::abs(static_cast<int>(decoded[ch]) - static_cast<int>(color[ch])) <=
                           2);
        }
    }
}

CG_TEST(bc4_six_value_mode_keeps_exact_extremes)
{
    // A block with exact black, exact white, and mid greys: the six-value mode
    // stores 0 and 255 for free, so both survive bit-exactly.
    u8 red[16] = {0, 255, 100, 110, 0, 255, 120, 105, 0, 255, 115, 100, 0, 255, 120, 110};
    Byte block[8];
    u8 decoded[16];
    cyber::assets::EncodeBc4Block(red, block, BcEncodeSettings{});
    cyber::assets::DecodeBc4Block(block, decoded);
    for (u32 texel = 0; texel < 16; ++texel)
    {
        if (red[texel] == 0 || red[texel] == 255)
        {
            CG_EXPECT_EQ(decoded[texel], red[texel]);
        }
        else
        {
            CG_EXPECT_TRUE(std::abs(static_cast<int>(decoded[texel]) - red[texel]) <= 3);
        }
    }
}

CG_TEST(bc1_punch_through_alpha_is_preserved)
{
    u8 rgba[64];
    for (u32 texel = 0; texel < 16; ++texel)
    {
        rgba[texel * 4 + 0] = static_cast<u8>(texel * 16);
        rgba[texel * 4 + 1] = 90;
        rgba[texel * 4 + 2] = static_cast<u8>(255 - texel * 16);
        rgba[texel * 4 + 3] = (texel % 3 == 0) ? 0 : 255;
    }
    Byte block[8];
    u8 decoded[64];
    cyber::assets::EncodeBc1Block(rgba, block, BcEncodeSettings{});
    cyber::assets::DecodeBc1Block(block, decoded);
    for (u32 texel = 0; texel < 16; ++texel)
    {
        CG_EXPECT_EQ(decoded[texel * 4 + 3], rgba[texel * 4 + 3] == 0 ? 0u : 255u);
    }
}

CG_TEST(encoder_psnr_meets_the_quality_floor)
{
    HeapAllocator allocator;
    const TextureAsset card = MakeTestCard(allocator, false);
    const f64 bc1 = RoundTripPsnr(card, TextureFormat::Bc1Rgba, BcQuality::Normal, 0x7u, allocator);
    const f64 bc4 = RoundTripPsnr(card, TextureFormat::Bc4R, BcQuality::Normal, 0x1u, allocator);
    const f64 bc5 = RoundTripPsnr(card, TextureFormat::Bc5Rg, BcQuality::Normal, 0x3u, allocator);
    const f64 bc7Fast =
        RoundTripPsnr(card, TextureFormat::Bc7Rgba, BcQuality::Fast, 0xFu, allocator);
    const f64 bc7Normal =
        RoundTripPsnr(card, TextureFormat::Bc7Rgba, BcQuality::Normal, 0xFu, allocator);
    const f64 bc7High =
        RoundTripPsnr(card, TextureFormat::Bc7Rgba, BcQuality::High, 0xFu, allocator);
    CG_EXPECT_TRUE(bc1 >= kMinPsnrBc1);
    CG_EXPECT_TRUE(bc4 >= kMinPsnrBc4);
    CG_EXPECT_TRUE(bc5 >= kMinPsnrBc5);
    CG_EXPECT_TRUE(bc7Fast >= kMinPsnrBc7Fast);
    CG_EXPECT_TRUE(bc7Normal >= kMinPsnrBc7Normal);
    // More search never makes the result worse.
    CG_EXPECT_TRUE(bc7Normal >= bc7Fast - 0.01);
    CG_EXPECT_TRUE(bc7High >= bc7Normal - 0.01);
    // BC7 is the high-quality format; it must beat BC1 on colour.
    CG_EXPECT_TRUE(bc7Normal > bc1);
}

CG_TEST(bc7_alpha_blocks_meet_the_quality_floor)
{
    HeapAllocator allocator;
    const TextureAsset card = MakeTestCard(allocator, true);
    const f64 psnr =
        RoundTripPsnr(card, TextureFormat::Bc7Rgba, BcQuality::Normal, 0xFu, allocator);
    CG_EXPECT_TRUE(psnr >= kMinPsnrBc7Alpha);
}

CG_TEST(encoding_is_deterministic)
{
    HeapAllocator allocator;
    const TextureAsset card = MakeTestCard(allocator, true);
    for (const TextureFormat format : {TextureFormat::Bc1Rgba, TextureFormat::Bc2Rgba,
                                       TextureFormat::Bc3Rgba, TextureFormat::Bc7Rgba})
    {
        auto first = CompressTexture(card, format, BcEncodeSettings{}, allocator);
        auto second = CompressTexture(card, format, BcEncodeSettings{}, allocator);
        CG_REQUIRE(first.IsOk() && second.IsOk());
        CG_REQUIRE(first.Value().data.Size() == second.Value().data.Size());
        CG_EXPECT_EQ(std::memcmp(first.Value().data.Data(), second.Value().data.Data(),
                                 first.Value().data.Size()),
                     0);
    }
}

CG_TEST(mip_chain_and_odd_sizes_compress_with_a_valid_layout)
{
    HeapAllocator allocator;
    TextureAsset source(allocator);
    source.format = TextureFormat::Rgba8Unorm;
    source.width = 13;
    source.height = 7;
    (void)source.data.TryResize(13 * 7 * 4);
    (void)source.levels.TryResize(1);
    source.levels[0] = cyber::assets::TextureLevel{0, 13 * 7 * 4, 13, 7, 13 * 4};
    for (u32 i = 0; i < 13 * 7 * 4; ++i)
    {
        source.data[i] = static_cast<Byte>((i * 37u) & 0xFFu);
    }

    auto chain = GenerateMipChain(source, true, allocator);
    CG_REQUIRE(chain.IsOk());
    CG_EXPECT_EQ(chain.Value().mipCount, 4u);  // 13x7, 6x3, 3x1, 1x1
    CG_EXPECT_EQ(chain.Value().levels[3].width, 1u);
    CG_EXPECT_TRUE(chain.Value().Validate().IsOk());

    auto compressed =
        CompressTexture(chain.Value(), TextureFormat::Bc7Rgba, BcEncodeSettings{}, allocator);
    CG_REQUIRE(compressed.IsOk());
    CG_EXPECT_TRUE(compressed.Value().Validate().IsOk());
    // 13x7 rounds up to 4x2 blocks of 16 bytes.
    CG_EXPECT_EQ(compressed.Value().levels[0].size, 4u * 2u * 16u);
    auto decoded = DecompressTexture(compressed.Value(), allocator);
    CG_REQUIRE(decoded.IsOk());
    CG_EXPECT_EQ(decoded.Value().levels[1].width, 6u);
    CG_EXPECT_EQ(decoded.Value().levels[1].height, 3u);
}

CG_TEST(srgb_mips_average_in_linear_light)
{
    HeapAllocator allocator;
    TextureAsset source(allocator);
    source.format = TextureFormat::Rgba8Unorm;
    source.width = 2;
    source.height = 2;
    (void)source.data.TryResize(16);
    (void)source.levels.TryResize(1);
    source.levels[0] = cyber::assets::TextureLevel{0, 16, 2, 2, 8};
    const u8 texels[16] = {0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255};
    std::memcpy(source.data.Data(), texels, 16);

    auto linear = GenerateMipChain(source, false, allocator);
    auto srgb = GenerateMipChain(source, true, allocator);
    CG_REQUIRE(linear.IsOk() && srgb.IsOk());
    const auto top = [](const TextureAsset& texture)
    {
        return static_cast<u32>(
            texture.data[static_cast<cyber::core::usize>(texture.levels[1].offset)]);
    };
    CG_EXPECT_EQ(top(linear.Value()), 128u);
    // Half-intensity light is sRGB 188, not 128.
    CG_EXPECT_EQ(top(srgb.Value()), 188u);
}

CG_TEST(psnr_reports_identical_and_unusable_inputs)
{
    const u8 a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const u8 b[8] = {1, 2, 3, 4, 5, 6, 7, 9};
    CG_EXPECT_TRUE(std::isinf(ComputePsnr(Span<const u8>(a, 8), Span<const u8>(a, 8), 2, 1)));
    CG_EXPECT_TRUE(std::isinf(ComputePsnr(Span<const u8>(a, 8), Span<const u8>(b, 8), 2, 1, 0x7u)));
    CG_EXPECT_TRUE(ComputePsnr(Span<const u8>(a, 8), Span<const u8>(b, 8), 2, 1, 0xFu) > 50.0);
    CG_EXPECT_EQ(ComputePsnr(Span<const u8>(a, 8), Span<const u8>(b, 8), 4, 4), 0.0);
}

CG_TEST(bc2_and_bc3_decode_the_colour_block_in_four_colour_mode)
{
    // Endpoints in "wrong" order (c0 < c1) would select BC1's three-colour mode;
    // BC2 and BC3 ignore the order, so index 2 is (2 * c0 + c1) / 3 = 85.
    const Byte colour[8] = {Byte{0x00}, Byte{0x00}, Byte{0xFF}, Byte{0xFF},
                            Byte{0xAA}, Byte{0xAA}, Byte{0xAA}, Byte{0xAA}};
    Byte bc2[16] = {};
    Byte bc3[16] = {};
    std::memcpy(bc2 + 8, colour, 8);
    std::memcpy(bc3 + 8, colour, 8);
    // BC2 alpha: texel 0 = 1, texel 1 = 15, texel 2 = 0, texel 3 = 8.
    bc2[0] = Byte{0xF1};
    bc2[1] = Byte{0x80};
    // BC3 alpha: both endpoints 200 gives a flat 200 for any index.
    bc3[0] = Byte{200};
    bc3[1] = Byte{200};
    u8 decoded2[64];
    u8 decoded3[64];
    cyber::assets::DecodeBc2Block(bc2, decoded2);
    cyber::assets::DecodeBc3Block(bc3, decoded3);
    for (u32 texel = 0; texel < 16; ++texel)
    {
        for (u32 ch = 0; ch < 3; ++ch)
        {
            CG_EXPECT_EQ(decoded2[texel * 4 + ch], 85u);
            CG_EXPECT_EQ(decoded3[texel * 4 + ch], 85u);
        }
        CG_EXPECT_EQ(decoded3[texel * 4 + 3], 200u);
    }
    CG_EXPECT_EQ(decoded2[3], 17u);
    CG_EXPECT_EQ(decoded2[7], 255u);
    CG_EXPECT_EQ(decoded2[11], 0u);
    CG_EXPECT_EQ(decoded2[15], 136u);
}

CG_TEST(bc2_and_bc3_encoders_meet_the_quality_floor)
{
    HeapAllocator allocator;
    const TextureAsset card = MakeTestCard(allocator, true);
    const f64 bc2 = RoundTripPsnr(card, TextureFormat::Bc2Rgba, BcQuality::Normal, 0xFu, allocator);
    const f64 bc3 = RoundTripPsnr(card, TextureFormat::Bc3Rgba, BcQuality::Normal, 0xFu, allocator);
    const TextureAsset opaque = MakeTestCard(allocator, false);
    const f64 bc3Colour =
        RoundTripPsnr(opaque, TextureFormat::Bc3Rgba, BcQuality::Normal, 0x7u, allocator);
    // BC3 carries the same colour data as BC1, so opaque colour quality must
    // stay within about 1 dB of the BC1 floor.
    CG_EXPECT_TRUE(bc3Colour >= kMinPsnrBc1 - 1.0);
    CG_EXPECT_TRUE(bc2 >= kMinPsnrBc2);
    CG_EXPECT_TRUE(bc3 >= kMinPsnrBc3);
    // BC3's interpolated alpha beats BC2's 4-bit alpha on this gradient.
    CG_EXPECT_TRUE(bc3 > bc2);
}

CG_TEST(bc2_and_bc3_solid_and_edge_blocks_decode_to_what_was_scored)
{
    const BcEncodeSettings settings;
    for (const int alphaValue : {0, 77, 255})
    {
        const u8 alpha = static_cast<u8>(alphaValue);
        u8 rgba[64];
        for (u32 texel = 0; texel < 16; ++texel)
        {
            rgba[texel * 4] = 40;
            rgba[texel * 4 + 1] = 160;
            rgba[texel * 4 + 2] = 220;
            rgba[texel * 4 + 3] = alpha;
        }
        Byte block[16];
        u8 decoded[64];
        cyber::assets::EncodeBc3Block(rgba, block, settings);
        cyber::assets::DecodeBc3Block(block, decoded);
        for (u32 texel = 0; texel < 16; ++texel)
        {
            for (u32 ch = 0; ch < 3; ++ch)
            {
                CG_EXPECT_TRUE(std::abs(static_cast<int>(decoded[texel * 4 + ch]) -
                                        static_cast<int>(rgba[texel * 4 + ch])) <= 2);
            }
            CG_EXPECT_EQ(decoded[texel * 4 + 3], alpha);
        }
        cyber::assets::EncodeBc2Block(rgba, block, settings);
        cyber::assets::DecodeBc2Block(block, decoded);
        for (u32 texel = 0; texel < 16; ++texel)
        {
            CG_EXPECT_TRUE(
                std::abs(static_cast<int>(decoded[texel * 4 + 3]) - static_cast<int>(alpha)) <= 8);
        }
    }
}

CG_TEST(bc7_alpha_quality_floors_hold_and_more_search_never_hurts)
{
    HeapAllocator allocator;
    const TextureAsset card = MakeTestCard(allocator, true);
    const f64 fast = RoundTripPsnr(card, TextureFormat::Bc7Rgba, BcQuality::Fast, 0xFu, allocator);
    const f64 normal =
        RoundTripPsnr(card, TextureFormat::Bc7Rgba, BcQuality::Normal, 0xFu, allocator);
    const f64 high = RoundTripPsnr(card, TextureFormat::Bc7Rgba, BcQuality::High, 0xFu, allocator);
    CG_EXPECT_TRUE(normal >= fast - 0.01);
    CG_EXPECT_TRUE(high >= normal - 0.01);
    CG_EXPECT_TRUE(normal >= kMinPsnrBc7AlphaNormal);
    CG_EXPECT_TRUE(high >= kMinPsnrBc7AlphaHigh);
}
