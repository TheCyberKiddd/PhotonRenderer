#pragma once

#include <cyber/assets/asset.h>
#include <cyber/assets/texture.h>
#include <cyber/core/types.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/span.h>

namespace cyber::assets
{

using cyber::core::f32;
using cyber::core::f64;
using cyber::core::u16;
using cyber::core::u32;
using cyber::core::u8;

/// Block-compression (BCn) codec: decoders for BC1 to BC5, BC6H and BC7, which
/// define the reference output, and encoders for the same formats.
///
/// Every block function works on one 4x4 block of texels in row-major order.
/// RGBA blocks are 64 bytes (16 texels x RGBA8), BC4 input is 16 single-channel
/// bytes, and BC5 input is 32 bytes of interleaved red/green pairs.
///
/// Decoders follow the Direct3D 11 functional specification. BC1 and BC4
/// interpolate in integer arithmetic with round-to-nearest; BC7 uses the exact
/// 6-bit weight interpolation the specification mandates, so its output is
/// bit-exact with any conforming decoder. Encoders are deterministic: the same
/// input and settings always produce the same bytes on every platform.

/// Search effort. Higher levels try more endpoint refinements, and for BC7
/// more partitions and modes; quality never decreases with the level.
enum class BcQuality : u8
{
    Fast = 0,
    Normal = 1,
    High = 2,
};

struct BcEncodeSettings
{
    BcQuality quality = BcQuality::Normal;
    /// BC1 texels with alpha below this value are encoded transparent using the
    /// three-colour block mode. Zero disables punch-through alpha.
    u8 bc1AlphaThreshold = 128;
};

inline constexpr usize kBcBlockTexels = 16;
inline constexpr usize kBcRgbaBlockBytes = kBcBlockTexels * 4;

// ---- Block decoders ---------------------------------------------------------

void DecodeBc1Block(const Byte* block, u8* rgba) noexcept;
/// Explicit 4-bit alpha (value * 17) and an always-four-colour BC1 colour block.
void DecodeBc2Block(const Byte* block, u8* rgba) noexcept;
/// A BC4 alpha block and an always-four-colour BC1 colour block.
void DecodeBc3Block(const Byte* block, u8* rgba) noexcept;
/// Writes 16 red values.
void DecodeBc4Block(const Byte* block, u8* red) noexcept;
/// Writes 16 interleaved red/green pairs.
void DecodeBc5Block(const Byte* block, u8* redGreen) noexcept;
/// Reserved mode bytes (a zero first byte) decode to transparent black, as the
/// specification requires.
void DecodeBc7Block(const Byte* block, u8* rgba) noexcept;

// ---- Block encoders ---------------------------------------------------------

void EncodeBc1Block(const u8* rgba, Byte* block, const BcEncodeSettings& settings) noexcept;
/// Alpha rounded to 4 bits; colour as in BC3.
void EncodeBc2Block(const u8* rgba, Byte* block, const BcEncodeSettings& settings) noexcept;
/// Alpha as a BC4 block; colour from the BC1 encoder with every index re-picked
/// against the four-colour palette BC3 always decodes with.
void EncodeBc3Block(const u8* rgba, Byte* block, const BcEncodeSettings& settings) noexcept;
void EncodeBc4Block(const u8* red, Byte* block, const BcEncodeSettings& settings) noexcept;
void EncodeBc5Block(const u8* redGreen, Byte* block, const BcEncodeSettings& settings) noexcept;
/// Opaque blocks search modes 6, 1 and 3 (and modes 0 and 2 at High); blocks
/// with alpha search modes 6, 5, 4 (not at Fast) and 7 (not at Fast); High also
/// tries the four channel rotations of modes 4 and 5. Partitioned modes test the best-scoring
/// partitions from a cheap per-subset line-fit estimate.
void EncodeBc7Block(const u8* rgba, Byte* block, const BcEncodeSettings& settings) noexcept;

// ---- BC6H (HDR) -------------------------------------------------------------
//
// BC6H stores RGB as half-float endpoints (no alpha). Texels travel as IEEE 754
// half-float bit patterns, three values (R, G, B) per texel, 48 per block, so a
// caller converts float data with `FloatToHalf`. The decoder is bit-exact with
// the Direct3D 11 specification for both the unsigned (UF16) and signed (SF16)
// variants. The encoder picks among all 14 modes; unsigned encoding clamps
// negative values to zero, and both clamp to the largest finite half.
// `TextureFormat::Bc6hRgb` is the unsigned variant.

/// IEEE 754 binary16 to binary32 (exact), and back with round-to-nearest-even,
/// overflowing to infinity.
[[nodiscard]] f32 HalfToFloat(u16 half) noexcept;
[[nodiscard]] u16 FloatToHalf(f32 value) noexcept;

inline constexpr usize kBc6hBlockBytes = 16;
inline constexpr usize kBc6hBlockValues = kBcBlockTexels * 3;

/// Reserved modes decode to zero, as the specification requires.
void DecodeBc6hBlock(const Byte* block, u16* rgbHalf, bool isSigned = false) noexcept;
/// Fast searches the single-subset modes 11 and 12; Normal adds modes 13 and
/// 14, the best four partitions of the two-subset modes and a polish of the best
/// two candidates; High tries all 32 partitions and polishes the best four.
void EncodeBc6hBlock(const u16* rgbHalf, Byte* block, const BcEncodeSettings& settings,
                     bool isSigned = false) noexcept;

/// Compressed size of a `width x height` image, in bytes.
[[nodiscard]] constexpr usize Bc6hSurfaceBytes(u32 width, u32 height) noexcept
{
    return static_cast<usize>((width + 3u) / 4u) * ((height + 3u) / 4u) * kBc6hBlockBytes;
}

/// Encodes `width x height` texels, four half values (R, G, B, ignored) per
/// texel, row-major, into `blocks` (`Bc6hSurfaceBytes` bytes). Edge blocks
/// replicate the last row and column. Returns false on a size mismatch.
[[nodiscard]] bool CompressBc6hSurface(Span<const u16> rgbaHalf, u32 width, u32 height,
                                       Span<Byte> blocks, const BcEncodeSettings& settings,
                                       bool isSigned = false) noexcept;
/// The inverse: writes four half values per texel, alpha 1.0. Returns false on a
/// size mismatch.
[[nodiscard]] bool DecompressBc6hSurface(Span<const Byte> blocks, u32 width, u32 height,
                                         Span<u16> rgbaHalf, bool isSigned = false) noexcept;

// ---- Surfaces ---------------------------------------------------------------

/// True for the formats this codec can both encode and decode.
[[nodiscard]] constexpr bool IsBcCodecFormat(TextureFormat format) noexcept
{
    return format == TextureFormat::Bc1Rgba || format == TextureFormat::Bc2Rgba ||
           format == TextureFormat::Bc3Rgba || format == TextureFormat::Bc4R ||
           format == TextureFormat::Bc5Rg || format == TextureFormat::Bc7Rgba;
}

/// Encodes every level and layer of an Rgba8Unorm texture. BC4 encodes the red
/// channel and BC5 red and green. Edge blocks of levels that are not a
/// multiple of four replicate the last row and column. The block encoders are
/// reentrant, so callers that need parallelism can split a surface into block
/// rows and call them from several threads.
[[nodiscard]] ImportResult<TextureAsset> CompressTexture(const TextureAsset& source,
                                                         TextureFormat target,
                                                         const BcEncodeSettings& settings,
                                                         Allocator& allocator) noexcept;

/// Decodes every level and layer of a BC1/BC4/BC5/BC7 texture to Rgba8Unorm.
/// BC4 decodes to (r, 0, 0, 255) and BC5 to (r, g, 0, 255).
[[nodiscard]] ImportResult<TextureAsset> DecompressTexture(const TextureAsset& source,
                                                           Allocator& allocator) noexcept;

/// Replaces the level chain of a single-level Rgba8Unorm texture with a full
/// chain down to 1x1. Each level is a 2x2 box filter of the previous one; an
/// odd dimension folds its last column or row into the final texel (a 3-tap
/// box there), so no source texel is dropped. With `srgb` the colour channels
/// are filtered in linear light and re-encoded; alpha is always filtered
/// linearly.
[[nodiscard]] ImportResult<TextureAsset> GenerateMipChain(const TextureAsset& source, bool srgb,
                                                          Allocator& allocator) noexcept;

/// Peak signal-to-noise ratio in dB between two RGBA8 images of the same size,
/// over the channels set in `channelMask` (bit 0 red .. bit 3 alpha). Returns
/// +infinity for identical images and 0 when the inputs are unusable.
[[nodiscard]] f64 ComputePsnr(Span<const u8> lhs, Span<const u8> rhs, u32 width, u32 height,
                              u32 channelMask = 0x7u) noexcept;

}  // namespace cyber::assets
