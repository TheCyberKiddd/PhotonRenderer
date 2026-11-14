#include <cyber/software/debug_text.h>
#include <cyber/software/denoise.h>
#include <cyber/software/frame_pacer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/gi_quality.h>
#include <cyber/software/global_illumination.h>
#include <cyber/software/image_io.h>
#include <cyber/software/present_target.h>
#include <cyber/software/raster.h>
#include <cyber/software/raytrace.h>
#include <cyber/software/software.h>
#include <cyber/test/test.h>

#include <type_traits>

using namespace cyber::software;

CG_TEST(software_public_headers)
{
    static_assert(!std::is_copy_constructible_v<Framebuffer>);
    static_assert(!std::is_copy_assignable_v<Framebuffer>);
    static_assert(std::is_move_constructible_v<Framebuffer>);
    static_assert(std::is_move_assignable_v<Framebuffer>);

    static_assert(PixelSizeBytes(PixelFormat::RGBA8Unorm) == 4);
    static_assert(PixelSizeBytes(PixelFormat::RGBA32F) == 16);
    static_assert(!IsValidPixelFormat(static_cast<PixelFormat>(42)));

    static_assert(kGlyphWidth == 5);
    static_assert(kGlyphHeight == 7);
    static_assert(kGlyphAdvance == 6);

    FramebufferDesc desc{};
    CG_EXPECT_EQ(desc.width, 0);
    CG_EXPECT_EQ(desc.height, 0);
    CG_EXPECT_EQ(desc.strideBytes, usize{0});

    FramePacer pacer;
    CG_EXPECT_EQ(pacer.TargetFramesPerSecond(), cyber::core::u32{60});

    RasterizerDesc rasterDesc{};
    CG_EXPECT_EQ(rasterDesc.tileSize, i32{64});
    CG_EXPECT_TRUE(rasterDesc.useJobs);
    RasterStats rasterStats{};
    CG_EXPECT_EQ(rasterStats.pixels, u64{0});

    CG_EXPECT_EQ(PresentByteSize(3, 2), usize{24});
    CG_EXPECT_TRUE(MeasureText("A").width == kGlyphWidth);
    CG_EXPECT_TRUE(IsGlyphSupported('7'));
    CG_EXPECT_FALSE(IsGlyphSupported('a'));
}

// Phase 31 headers stand alone and keep their ownership contracts.
CG_TEST(software_public_headers_phase31)
{
    static_assert(!std::is_copy_constructible_v<BottomLevelAccelerationStructure>);
    static_assert(std::is_nothrow_move_constructible_v<BottomLevelAccelerationStructure>);
    static_assert(!std::is_copy_constructible_v<TopLevelAccelerationStructure>);
    static_assert(std::is_nothrow_move_constructible_v<TopLevelAccelerationStructure>);
    static_assert(!std::is_copy_constructible_v<IrradianceVolume>);
    static_assert(std::is_nothrow_move_constructible_v<IrradianceVolume>);
    static_assert(!std::is_copy_constructible_v<SpatioTemporalDenoiser>);
    static_assert(std::is_nothrow_move_constructible_v<SpatioTemporalDenoiser>);
    static_assert(std::is_trivially_copyable_v<RayDesc>);
    static_assert(std::is_trivially_copyable_v<RayHit>);
    static_assert(std::is_trivially_copyable_v<IrradianceProbe>);

    const BottomLevelAccelerationStructure blas;
    CG_EXPECT_FALSE(blas.IsValid());
    CG_EXPECT_TRUE(blas.Bounds().IsEmpty());
    const IrradianceVolume volume;
    CG_EXPECT_FALSE(volume.IsValid());
    CG_EXPECT_EQ(volume.Probe(0).state, IrradianceProbeState::Unbaked);
    const SpatioTemporalDenoiser denoiser;
    CG_EXPECT_FALSE(denoiser.IsValid());
}
