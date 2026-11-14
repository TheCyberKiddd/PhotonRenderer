#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/core/types.h>
#include <cyber/math/color.h>
#include <cyber/math/mat4.h>
#include <cyber/math/vec2.h>
#include <cyber/math/vec3.h>
#include <cyber/memory/allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>

namespace cyber::jobs
{
class JobSystem;
}

namespace cyber::software
{

using cyber::core::f32;
using cyber::core::i32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;
using cyber::core::usize;

// The HDR post-processing chain of the software reference renderer.
//
// The renderer shades into an unbounded linear HDR target. Everything in this
// header runs after shading and before the image is encoded for display:
// ambient occlusion, depth of field, motion blur, bloom, exposure, tone
// mapping, and color grading. Each stage is a separate pass with explicit
// inputs, each can be disabled on its own, and each reports the time it cost,
// so a caller can price the chain stage by stage on the machine it ships on.
//
// Units and conventions, which the whole chain shares:
//
//  - Color is linear RGB, unbounded above 1 until the tone mapper runs. No
//    transfer function is applied here; present or export encodes sRGB.
//  - Luminance is the Rec. 709 luma of linear RGB, in the same arbitrary
//    radiometric unit the shading path produces.
//  - Depth is the renderer's reverse-Z convention: 1 at the near plane, 0 at
//    the far plane. LinearizeReverseZ converts it to a view-space distance.
//  - Velocity is screen-space pixels per frame, x right and y down, stored in
//    the red and green channels of an RGBA32F image.
//  - Normals are world-space unit vectors stored in the red, green, and blue
//    channels, remapped to [0, 1]; the view matrix in PostProcessView brings
//    them into the view space the occlusion pass works in. That space has +Z
//    along the view direction, so a surface facing the camera has a view-space
//    normal of (0, 0, -1), which encodes as (0.5, 0.5, 0).
//
// Every pass is deterministic. Sample patterns come from fixed low-discrepancy
// sequences rather than a random source, and the job-parallel and
// single-threaded paths write identical bytes, which is what lets the
// acceptance tests assert on exact numbers.

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Rec. 709 luminance of a linear RGB color. Negative inputs are allowed and
// produce a negative result; the caller decides whether that is meaningful.
[[nodiscard]] f32 Luminance(const math::Color& color) noexcept;

// Converts a reverse-Z depth sample to a positive view-space distance.
// depth = 1 returns nearPlane and depth = 0 returns farPlane. Values outside
// [0, 1] clamp, and a non-finite depth returns farPlane, so a cleared texel
// reads as the far plane rather than poisoning the pass that consumes it.
[[nodiscard]] f32 LinearizeReverseZ(f32 depth, f32 nearPlane, f32 farPlane) noexcept;

// What the post chain needs to know about the camera that produced the frame.
struct PostProcessView
{
    // The unjittered reverse-Z projection the frame was rendered with. Only
    // the two diagonal scale terms are read, to recover the field of view.
    math::Mat4 projection = math::Mat4::Identity();
    // World to view. Used to bring world-space normals into view space.
    math::Mat4 view = math::Mat4::Identity();
    f32 nearPlane = 0.1f;
    f32 farPlane = 1000.0f;
    // Sensor height used to convert a circle of confusion in millimetres to
    // pixels. 24 mm is the full-frame 35 mm height.
    f32 sensorHeightMm = 24.0f;
};

// Field-of-view half-extents at unit view distance, recovered from a
// projection matrix, plus the depth range. Derived once per frame and passed
// down to the passes that unproject.
struct ViewGeometry
{
    f32 tanHalfFovX = 1.0f;
    f32 tanHalfFovY = 1.0f;
    f32 nearPlane = 0.1f;
    f32 farPlane = 1000.0f;
};

// Recovers the view geometry from view.projection. A degenerate or non-finite
// projection term falls back to a 45 degree half-angle, so an unset camera
// produces a usable, if wrong, frustum instead of NaNs.
[[nodiscard]] ViewGeometry MakeViewGeometry(const PostProcessView& view) noexcept;

// View-space position of a pixel center, with +Z pointing away from the camera
// along the view direction. linearDepth is the value LinearizeReverseZ returns.
[[nodiscard]] math::Vec3 ReconstructViewPosition(i32 x, i32 y, i32 width, i32 height,
                                                 f32 linearDepth,
                                                 const ViewGeometry& geometry) noexcept;

// ---------------------------------------------------------------------------
// Exposure
// ---------------------------------------------------------------------------

enum class ExposureMode : u8
{
    // The scene is scaled by manualEv alone.
    Manual = 0,
    // The scene is scaled so its measured average luminance lands on middle
    // grey, then offset by compensationEv.
    Automatic,
};

inline constexpr i32 kLuminanceHistogramBins = 64;
// Middle grey. An automatic exposure maps the measured average luminance here.
inline constexpr f32 kMiddleGrey = 0.18f;

// A log2-luminance histogram of one frame.
//
// Bins cover [minLogLuminance, maxLogLuminance] in log2 space. Pixels below the
// range land in bin 0 and pixels above it in the last bin, so no energy is
// lost and the percentile walk stays total-preserving.
struct LuminanceHistogram
{
    u32 bins[kLuminanceHistogramBins]{};
    u32 totalSamples = 0;
    f32 minLogLuminance = -10.0f;
    f32 maxLogLuminance = 10.0f;
};

struct ExposureSettings
{
    ExposureMode mode = ExposureMode::Automatic;
    // Applied in Manual mode: the scene is multiplied by exp2(manualEv).
    f32 manualEv = 0.0f;
    // Applied on top of the metered value in Automatic mode.
    f32 compensationEv = 0.0f;
    // Bounds on the metered exposure, before compensation. A dark or blown-out
    // frame cannot drive the camera past these.
    f32 minEv = -8.0f;
    f32 maxEv = 8.0f;
    // Histogram percentiles that define the average. Discarding the darkest
    // half and the brightest twentieth keeps a small bright window or a black
    // doorway from dragging the whole frame with it. lowPercent must be below
    // highPercent; both are fractions in [0, 1].
    f32 lowPercent = 0.50f;
    f32 highPercent = 0.95f;
    // Eye adaptation, in EV per second. Brightening is faster than darkening,
    // which is both what a real eye does and what hides a flicker.
    f32 adaptUpEvPerSecond = 3.0f;
    f32 adaptDownEvPerSecond = 1.0f;
    // Luminance range the histogram covers, in log2 units.
    f32 minLogLuminance = -10.0f;
    f32 maxLogLuminance = 10.0f;
};

// Carries the adapted exposure between frames. The first frame snaps to its
// metered value instead of ramping from zero.
struct ExposureState
{
    f32 currentEv = 0.0f;
    bool adapted = false;
};

// Builds the log2-luminance histogram of an HDR image. Non-finite and
// non-positive pixels are counted in bin 0 rather than skipped, so totalSamples
// always equals the pixel count. Returns InvalidArgument for an invalid image
// or an empty log range.
[[nodiscard]] core::Status<core::Error> BuildLuminanceHistogram(const Framebuffer& hdrColor,
                                                                f32 minLogLuminance,
                                                                f32 maxLogLuminance,
                                                                LuminanceHistogram& out) noexcept;

// Average luminance of the samples between the two percentiles. An empty
// histogram or an inverted percentile window returns kMiddleGrey, which makes
// the exposure a no-op rather than a jump to black.
[[nodiscard]] f32 HistogramAverageLuminance(const LuminanceHistogram& histogram, f32 lowPercent,
                                            f32 highPercent) noexcept;

// EV that maps averageLuminance onto middle grey: -log2(average / 0.18).
// A non-positive or non-finite average returns 0.
[[nodiscard]] f32 ExposureEvForAverageLuminance(f32 averageLuminance) noexcept;

// Advances state toward targetEv at the configured rate and returns the EV to
// render with. The first call snaps. A non-positive or non-finite
// deltaSeconds also snaps, so a paused frame does not stall mid-ramp.
[[nodiscard]] f32 AdaptExposure(ExposureState& state, f32 targetEv, f32 deltaSeconds,
                                const ExposureSettings& settings) noexcept;

// The EV a frame should be rendered with, metering hdrColor when the mode is
// Automatic. Updates state. Returns InvalidArgument when metering is needed and
// hdrColor is invalid.
[[nodiscard]] core::Result<f32, core::Error> ComputeExposureEv(const Framebuffer& hdrColor,
                                                               const ExposureSettings& settings,
                                                               ExposureState& state,
                                                               f32 deltaSeconds) noexcept;

// ---------------------------------------------------------------------------
// Tone mapping and color grading
// ---------------------------------------------------------------------------

enum class ToneMapOperator : u8
{
    // Exposure and a clamp to [0, 1]. Kept as the control case: it is what the
    // other operators are measured against, and it clips instead of rolling
    // off.
    Clamp = 0,
    // Extended Reinhard with a white point. Gentle, desaturates highlights.
    Reinhard,
    // The ACES RRT and ODT fit. The renderer-wide default.
    AcesFitted,
    // Hable's filmic curve. Deeper toe, more contrast than ACES.
    Filmic,
};

inline constexpr ToneMapOperator kDefaultToneMapOperator = ToneMapOperator::AcesFitted;

struct ToneMapSettings
{
    ToneMapOperator op = kDefaultToneMapOperator;
    // Reinhard white point: the linear value that maps to 1. Ignored by the
    // other operators.
    f32 whitePoint = 4.0f;
};

// Applies exposure in stops, then the operator. The result is linear and
// clamped to [0, 1]; alpha passes through. Non-finite channels map to 0.
[[nodiscard]] math::Color ApplyToneMap(const math::Color& hdrColor, f32 exposureEv,
                                       const ToneMapSettings& settings) noexcept;

// Creative grade, evaluated in the display-referred [0, 1] domain the tone
// mapper produces. Defaults are the identity grade.
struct ColorGradeSettings
{
    // Lift raises the shadows, gain scales the highlights, and gamma is a
    // per-channel power applied between them. Neutral values are 0, 1, 1.
    math::Color lift{0.0f, 0.0f, 0.0f, 0.0f};
    math::Color gamma{1.0f, 1.0f, 1.0f, 1.0f};
    math::Color gain{1.0f, 1.0f, 1.0f, 1.0f};
    // Pivoted around middle grey, so contrast does not shift exposure.
    f32 contrast = 1.0f;
    // 0 is monochrome, 1 neutral; above 1 oversaturates.
    f32 saturation = 1.0f;
    // White balance. Positive temperature warms, positive tint pushes magenta.
    // Both are in [-1, 1].
    f32 temperature = 0.0f;
    f32 tint = 0.0f;
};

// Evaluates the grade on one display-referred color. Used directly by tests and
// to bake the lookup table below.
[[nodiscard]] math::Color ApplyColorGrade(const math::Color& color,
                                          const ColorGradeSettings& settings) noexcept;

inline constexpr i32 kMinColorGradeLutSize = 2;
inline constexpr i32 kMaxColorGradeLutSize = 64;
inline constexpr i32 kDefaultColorGradeLutSize = 32;

// A baked 3D color grading lookup table.
//
// The table is the grade sampled on a regular lattice over the unit RGB cube
// and reconstructed with trilinear interpolation, which is what a GPU does with
// a volume texture. Baking it once per settings change makes the per-pixel cost
// independent of how complex the grade is, and makes a grade authored in
// another tool loadable as data.
//
// An identity table reproduces its input exactly at the lattice points and to
// within the lattice spacing between them, so a disabled grade is stable.
class ColorGradeLut
{
public:
    ColorGradeLut() noexcept = default;
    ~ColorGradeLut();

    ColorGradeLut(const ColorGradeLut&) = delete;
    ColorGradeLut& operator=(const ColorGradeLut&) = delete;

    ColorGradeLut(ColorGradeLut&& other) noexcept;
    ColorGradeLut& operator=(ColorGradeLut&& other) noexcept;

    // size is the lattice edge, in [2, 64]; the table holds size^3 entries.
    // Returns InvalidArgument for a size out of range and Internal when the
    // allocation fails.
    [[nodiscard]] static core::Result<ColorGradeLut, core::Error>
    Bake(const ColorGradeSettings& settings, i32 size, memory::Allocator& allocator) noexcept;

    // The table that returns its input unchanged.
    [[nodiscard]] static core::Result<ColorGradeLut, core::Error>
    Identity(i32 size, memory::Allocator& allocator) noexcept;

    // Builds a table from externally authored entries, ordered with red
    // varying fastest and blue slowest. Returns InvalidArgument when the span
    // is not exactly size^3 entries.
    [[nodiscard]] static core::Result<ColorGradeLut, core::Error>
    FromEntries(memory::Span<const math::Color> entries, i32 size,
                memory::Allocator& allocator) noexcept;

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] i32 Size() const noexcept;
    [[nodiscard]] memory::Span<const math::Color> Entries() const noexcept;

    // Trilinear lookup. The input is clamped to [0, 1] per channel; alpha is
    // passed through untouched. An invalid table returns the input unchanged.
    [[nodiscard]] math::Color Sample(const math::Color& color) const noexcept;

private:
    void Release() noexcept;
    void MoveFrom(ColorGradeLut& other) noexcept;

    memory::Allocator* m_allocator = nullptr;
    math::Color* m_entries = nullptr;
    usize m_entryCount = 0;
    i32 m_size = 0;
};

// ---------------------------------------------------------------------------
// Per-effect settings
// ---------------------------------------------------------------------------

inline constexpr i32 kMaxBloomMips = 8;

struct BloomSettings
{
    // Luminance where bloom starts. The knee softens the transition so a
    // highlight does not pop in as it crosses the threshold.
    f32 threshold = 1.0f;
    f32 softKnee = 0.5f;
    // Fraction of the final image the bloom contributes, in [0, 1]. The blend
    // is a lerp, so energy is conserved rather than added.
    f32 intensity = 0.06f;
    // Downsample steps. Each one halves the resolution and widens the glow.
    i32 mipCount = 5;
    // Weight of the coarser mip at each upsample step, in [0, 1]. Higher
    // scatters the glow wider.
    f32 scatter = 0.7f;
};

struct AmbientOcclusionSettings
{
    // World-space hemisphere radius, in the scene's units.
    f32 radius = 0.5f;
    // Strength of the darkening. 0 is off, 1 is the raw occlusion term.
    f32 intensity = 1.0f;
    // View-space depth bias that keeps a flat surface from occluding itself.
    f32 bias = 0.02f;
    // Samples per pixel, clamped to [4, 64].
    i32 sampleCount = 16;
    // Radius, in pixels, of the depth-aware box blur that removes the sampling
    // noise. 0 disables the blur.
    i32 blurRadius = 2;
    // Exponent on the visibility term. Above 1 deepens contact shadows.
    f32 power = 1.0f;
};

struct DepthOfFieldSettings
{
    // Distance in view units that is perfectly sharp.
    f32 focusDistance = 5.0f;
    f32 focalLengthMm = 50.0f;
    // Aperture as an f-number. Smaller is a wider aperture and more blur.
    f32 apertureFStop = 2.8f;
    // Upper bound on the circle of confusion radius, in pixels. Bounds the
    // gather cost and keeps a far background from smearing the whole frame.
    f32 maxCocPixels = 12.0f;
    // Taps in the gather disc, clamped to [4, 64]. They are placed on a
    // golden-angle spiral, which is the cheapest pattern without visible
    // structure.
    i32 sampleCount = 24;
};

struct MotionBlurSettings
{
    // Fraction of the frame the shutter is open: 180 degrees is half a frame,
    // the film convention.
    f32 shutterAngleDegrees = 180.0f;
    // Taps along the velocity vector, clamped to [2, 32].
    i32 sampleCount = 8;
    // Clamp on the blur length, in pixels.
    f32 maxVelocityPixels = 32.0f;
};

// ---------------------------------------------------------------------------
// The stack
// ---------------------------------------------------------------------------

// Passes in execution order. The order is fixed because it is physical:
// occlusion and the lens effects act on scene-referred radiance, the camera
// response maps that to a display range, and the grade is a creative pass on
// the display-referred result.
enum class PostProcessPass : u8
{
    AmbientOcclusion = 0,
    DepthOfField,
    MotionBlur,
    Bloom,
    Exposure,
    ToneMap,
    ColorGrade,
};

inline constexpr i32 kPostProcessPassCount = 7;

// Stable identifier for a pass, for logs and on-screen readouts.
[[nodiscard]] const char* PostProcessPassName(PostProcessPass pass) noexcept;

struct PostProcessSettings
{
    // Per-effect runtime toggles. A disabled pass is skipped outright: it
    // reads nothing, writes nothing, and reports zero time.
    bool enableAmbientOcclusion = true;
    bool enableDepthOfField = false;
    bool enableMotionBlur = false;
    bool enableBloom = true;
    bool enableExposure = true;
    bool enableToneMap = true;
    bool enableColorGrade = false;

    ExposureSettings exposure{};
    ToneMapSettings toneMap{};
    BloomSettings bloom{};
    AmbientOcclusionSettings ambientOcclusion{};
    DepthOfFieldSettings depthOfField{};
    MotionBlurSettings motionBlur{};

    // Baked grade. Null with enableColorGrade set is not an error; the pass is
    // skipped and reported as disabled.
    const ColorGradeLut* colorGradeLut = nullptr;

    // False forces the single-threaded path. Both paths write identical bytes.
    bool useJobs = true;
};

// What one pass cost on this frame.
struct PassTiming
{
    const char* name = nullptr;
    // Wall time the pass took, measured on the CPU with the platform monotonic
    // clock. This renderer shades on the CPU, so this is the whole cost of the
    // effect, not a GPU queue timestamp.
    u64 nanoseconds = 0;
    // False when the pass was switched off or had nothing to do. A disabled
    // pass still appears in the table, so the readout does not reflow as
    // effects are toggled.
    bool executed = false;
};

struct PostProcessTimings
{
    PassTiming passes[kPostProcessPassCount]{};
    u64 totalNanoseconds = 0;
    // The EV the frame was tone mapped with, after metering and adaptation.
    f32 exposureEv = 0.0f;
    // Metered average luminance, or 0 when the frame was not metered.
    f32 averageLuminance = 0.0f;
};

// The buffers a frame hands to the chain. color is required; the rest are read
// only by the passes that need them, and a pass whose input is missing is
// skipped and reported as not executed rather than failing the frame.
struct PostProcessInputs
{
    const Framebuffer* color = nullptr;
    const DepthBuffer* depth = nullptr;
    // World-space normals encoded to [0, 1]. Required by ambient occlusion.
    const Framebuffer* normals = nullptr;
    // Screen-space pixels per frame in red and green. Required by motion blur.
    const Framebuffer* velocity = nullptr;
};

// Owns the scratch images the chain needs and runs the passes.
//
// The stack is sized once for a resolution and reused every frame: no pass
// allocates while a frame is in flight. Exposure state lives here too, because
// adaptation is a property of the camera over time, not of a single frame.
class PostProcessStack
{
public:
    PostProcessStack() noexcept = default;
    ~PostProcessStack() = default;

    PostProcessStack(const PostProcessStack&) = delete;
    PostProcessStack& operator=(const PostProcessStack&) = delete;

    PostProcessStack(PostProcessStack&&) noexcept = default;
    PostProcessStack& operator=(PostProcessStack&&) noexcept = default;

    // Allocates the ping-pong HDR pair, the occlusion buffer, and the bloom
    // chain for a width x height frame. Both dimensions must be in [1, 16384].
    // Returns InvalidArgument for a bad dimension and Internal when an
    // allocation fails.
    [[nodiscard]] static core::Result<PostProcessStack, core::Error>
    Create(i32 width, i32 height, memory::Allocator& allocator) noexcept;

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] i32 Width() const noexcept;
    [[nodiscard]] i32 Height() const noexcept;

    // Runs the enabled passes and writes the display-referred result into
    // destination, which must be valid and exactly Width() x Height().
    // destination may be any pixel format; an RGBA8Unorm target quantizes on
    // write, so a caller that wants to keep precision should pass RGBA32F.
    //
    // deltaSeconds drives exposure adaptation; pass the frame's own delta.
    // timings may be null. Returns InvalidArgument for a mismatched or invalid
    // buffer and Internal when a pass cannot complete.
    [[nodiscard]] core::Status<core::Error> Execute(const PostProcessInputs& inputs,
                                                    Framebuffer& destination,
                                                    const PostProcessSettings& settings,
                                                    const PostProcessView& view, f32 deltaSeconds,
                                                    jobs::JobSystem* jobSystem,
                                                    PostProcessTimings* timings) noexcept;

    // Exposure adaptation carried between frames. Reset it when the camera
    // cuts.
    [[nodiscard]] ExposureState& Exposure() noexcept;
    [[nodiscard]] const ExposureState& Exposure() const noexcept;

    // The occlusion factors the last frame produced, in the red channel, or an
    // invalid framebuffer when the pass has not run. Exposed so a debug view
    // can show the term that is modulating the image.
    [[nodiscard]] const Framebuffer& AmbientOcclusionBuffer() const noexcept;

private:
    [[nodiscard]] core::Status<core::Error>
    RunAmbientOcclusion(const PostProcessInputs& inputs, const PostProcessSettings& settings,
                        const PostProcessView& view, jobs::JobSystem* jobSystem) noexcept;
    [[nodiscard]] core::Status<core::Error> RunDepthOfField(const PostProcessInputs& inputs,
                                                            const PostProcessSettings& settings,
                                                            const PostProcessView& view,
                                                            jobs::JobSystem* jobSystem) noexcept;
    [[nodiscard]] core::Status<core::Error> RunMotionBlur(const PostProcessInputs& inputs,
                                                          const PostProcessSettings& settings,
                                                          jobs::JobSystem* jobSystem) noexcept;
    [[nodiscard]] core::Status<core::Error> RunBloom(const PostProcessSettings& settings,
                                                     jobs::JobSystem* jobSystem) noexcept;

    [[nodiscard]] Framebuffer& Current() noexcept;
    [[nodiscard]] Framebuffer& Scratch() noexcept;
    void Swap() noexcept;

    Framebuffer m_color[2];
    Framebuffer m_occlusion;
    Framebuffer m_bloomDown[kMaxBloomMips];
    Framebuffer m_bloomUp[kMaxBloomMips];
    ExposureState m_exposure{};
    memory::Allocator* m_allocator = nullptr;
    i32 m_width = 0;
    i32 m_height = 0;
    i32 m_bloomMipCount = 0;
    i32 m_currentIndex = 0;
    bool m_occlusionValid = false;
};

}  // namespace cyber::software
