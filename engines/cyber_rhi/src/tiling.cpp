#include <cyber/rhi/texture.h>

namespace cyber::rhi
{

namespace
{

// Standard 64 KiB tile shapes in elements (texels, or blocks for compressed
// formats) by element size. These are D3D12's standard swizzle shapes for 2D
// resources, which Vulkan's standard sparse image block shapes match.
bool StandardTileShape(u32 bytesPerElement, u32& width, u32& height) noexcept
{
    switch (bytesPerElement)
    {
    case 1:
        width = 256;
        height = 256;
        return true;
    case 2:
        width = 256;
        height = 128;
        return true;
    case 4:
        width = 128;
        height = 128;
        return true;
    case 8:
        width = 128;
        height = 64;
        return true;
    case 16:
        width = 64;
        height = 64;
        return true;
    default:
        return false;
    }
}

u32 CeilDiv(u32 value, u32 divisor) noexcept
{
    return (value + divisor - 1u) / divisor;
}

}  // namespace

TextureTiling ComputeStandardTiling(const TextureDesc& desc) noexcept
{
    TextureTiling tiling;
    if (desc.dimension != TextureDimension::Dim2D || desc.arrayLayers != 1 ||
        desc.sampleCount > 1 || desc.depth > 1 || desc.width == 0 || desc.height == 0 ||
        desc.mipLevels == 0 || desc.mipLevels > kMaxTiledMips ||
        static_cast<u32>(desc.format) >= kFormatCount || IsDepthFormat(desc.format))
    {
        return tiling;
    }
    const u32 bytes = FormatBytesPerBlock(desc.format);
    const u32 blockWidth = FormatBlockWidth(desc.format);
    const u32 blockHeight = FormatBlockHeight(desc.format);
    u32 shapeWidth = 0;
    u32 shapeHeight = 0;
    if (bytes == 0 || blockWidth == 0 || blockHeight == 0 ||
        !StandardTileShape(bytes, shapeWidth, shapeHeight))
    {
        return tiling;
    }
    tiling.tileWidth = shapeWidth * blockWidth;
    tiling.tileHeight = shapeHeight * blockHeight;

    // A mip is standard while it covers at least one whole tile in both
    // dimensions; from the first that does not, every smaller mip is packed.
    u64 packedBytes = 0;
    for (u32 mip = 0; mip < desc.mipLevels; ++mip)
    {
        const u32 width = desc.width >> mip > 0 ? desc.width >> mip : 1u;
        const u32 height = desc.height >> mip > 0 ? desc.height >> mip : 1u;
        const bool standard =
            tiling.packedMipCount == 0 && width >= tiling.tileWidth && height >= tiling.tileHeight;
        if (standard)
        {
            tiling.mipTilesX[mip] = CeilDiv(width, tiling.tileWidth);
            tiling.mipTilesY[mip] = CeilDiv(height, tiling.tileHeight);
            tiling.totalTileCount += tiling.mipTilesX[mip] * tiling.mipTilesY[mip];
            ++tiling.standardMipCount;
        }
        else
        {
            ++tiling.packedMipCount;
            packedBytes +=
                static_cast<u64>(CeilDiv(width, blockWidth)) * CeilDiv(height, blockHeight) * bytes;
        }
    }
    if (tiling.packedMipCount > 0)
    {
        tiling.packedTileCount =
            static_cast<u32>((packedBytes + kTileSizeBytes - 1) / kTileSizeBytes);
        tiling.totalTileCount += tiling.packedTileCount;
    }
    tiling.valid = true;
    return tiling;
}

}  // namespace cyber::rhi
