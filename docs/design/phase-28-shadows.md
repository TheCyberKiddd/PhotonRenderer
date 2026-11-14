# Phase 28 design — shadows

Date: 2026-11-09
Status: software PCSS/fog, stabilized cascades, depth-only map rendering,
D3D12 depth-map rendering, and measured software visual acceptance for acne,
detached shadows, cascade seams, and camera-motion shimmer implemented; GPU
scene integration remains out of scope for this phase
Related: [Phase 28 roadmap](../graphics/CYBERGRAPHICS_ROADMAP.md#phase-28-shadows), [build guide](../graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-28-shadows), [software renderer module contract](../../engines/cyber_software_renderer/docs/MODULE.md)

## Software shadow path

The software PBR renderer can consume caller-owned reverse-Z `DepthBuffer`
objects through `ShadowMapView`. The light's `clipFromWorld` matrix projects
each receiver into the map; the sampler performs bounded percentage-closer
filtering (PCF), with adjustable constant depth, slope-scaled depth, and
world-space normal-offset bias. Directional cascades, spot maps, and point-light
cube faces apply visibility only to direct contributions, leaving ambient
illumination unchanged. Lights without a shadow map retain their prior path and
avoid a shadow-sampling call.

The map uses the software renderer's existing reverse-Z convention: clear/far
depth is 0, near depth is 1, and a receiver is lit when its biased depth is
greater than or equal to the stored occluder depth. Out-of-frustum taps are lit.
Hard PCF radius is clamped to four texels (at most 81 depth comparisons per
shaded receiver); radius zero is a single comparison. Soft PCF uses a fixed
stratified 16-tap pattern and supports radii up to 16 texels without increasing
its comparison count. It avoids per-pixel random rotation but is not a
physically based area-light penumbra. Optional PCSS performs a bounded blocker
search, linearizes reverse-Z blocker and receiver depths, estimates penumbra
width from emitter size, then applies the same bounded 16-tap filter.
Perspective maps use a physical source radius; orthographic maps use an angular
source radius. Point-light visibility blends adjacent cube faces near seams to
reduce hard face-selection discontinuities. These CPU filters are quality-
oriented reference paths, not a GPU performance claim.

The software PBR path also provides default-off height fog with bounded
midpoint integration, height-dependent extinction, Henyey-Greenstein phase
scattering, and directional, point, and spot shadow queries along the view ray.
It is a per-pixel software feature, not a froxel/volumetric GPU implementation;
its cost grows with integration steps and shadowed lights.

`RenderDirectionalShadowMaps` is the end-to-end software cascade path. It
validates the cascade transforms, split order, and all caller-owned render
targets before starting; requires equal depth-map dimensions; renders all maps;
and only publishes the borrowed depth pointers after every map succeeds. This
avoids lighting with a partially refreshed cascade set after an allocation or
rasterization error. Target storage remains caller-owned and must outlive the
lighting data.

## Resources and cascade fitting

`RenderShadowMap` clears a caller-owned depth map and renders a mesh through a
depth-only clipped reverse-Z raster path. `DrawDepthOnlyTriangles` carries no
varyings through the near-plane clipper and `RasterizeDepthOnlyTriangles` takes
no color target and calls no pixel shader, so a shadow map costs coverage, the
depth test, and the depth store only. Coverage, bias handling, and the
resulting depth values are identical to the shaded path for the same geometry,
which the parity tests assert texel by texel and stat by stat; the trade is
that alpha-tested or shader-discarded occluders are not supported, because
there is no pixel stage. The RHI has owned create/import/destroy helpers for GPU shadow
depth resources, and depth-only graphics pipelines are supported. A D3D12
integration test rasterizes geometry through the render graph and reads back
the resulting depth value. `AddShadowSamplingPass` lets a consumer declare all
shadow-map reads in one graph pass, producing depth-write-to-shader-read
transitions and producer dependencies for an arbitrary map set. A Null-backend
integration test schedules two depth passes followed by a sampling consumer
and verifies both imported textures are in `ShaderRead` when the consumer
executes. This is resource scheduling infrastructure, not a production GPU
scene renderer.

Directional cascades use practical logarithmic/uniform split blending, a
rounded camera-frustum bounding sphere, padded light-space depth, and a
light-space center snapped to the shadow texel grid. A configurable overlap
blends visibility into the next cascade. The camera forward vector and
positive view depth are explicit inputs to PBR cascade selection.

Spot and point helpers generate reverse-Z projection transforms. Point PBR
sampling blends adjacent cube faces near seams. A deterministic fixed-tile CPU
atlas allocator supports power-of-two regions, guard bands, explicit release
with generation tokens, and a separate importance-to-resolution helper.
Cascade visualization colors the selected directional cascade in the software
PBR path.

## Measuring the acceptance bar

The roadmap's Phase 28 bar is a test scene with no shadow acne, no detached
shadows, and no visible seam between cascades. `shadow_quality.h` makes that
bar measurable instead of a matter of inspection. Each metric drives
caller-supplied probes through `EvaluateDirectionalShadowVisibility`, the same
sampler the PBR pixel shader calls, so a number describes the shipped path
rather than a parallel implementation. The functions allocate nothing, read
only borrowed data, and are deterministic. They define no scene and no
threshold: the caller supplies the probes and decides what passes.

- `MeasureShadowAcne` counts probes the caller knows are unoccluded that
  nevertheless sample shadowed, and reports the minimum, the mean, and the
  longest consecutive darkened run. A long run means a real occluder slipped
  into the probe set; acne is isolated speckle.
- `MeasureShadowContact` walks outward from an occluder's contact point and
  reports where visibility first reads lit, which is the detachment gap that
  excessive depth or normal-offset bias produces.
- `MeasureShadowTransect` reports the largest and mean visibility change
  between consecutive probes along a depth-ordered transect, and separately the
  largest change at a pair that straddles a cascade split.
- `MeasureShadowStability` analyzes a per-frame visibility sequence and reports
  the largest and mean frame-to-frame change plus the worst per-probe total
  variation, so a probe that flickers in small steps is still caught.

The acceptance scene is a ground plane and a long wall running along the camera
axis, lit by a directional light that casts the wall's shadow across the
ground. Because the wall's height and the light are constant, the ideal
visibility at a fixed distance from the shadow edge does not depend on depth.
A transect at such a fixed offset therefore has a constant ideal value, and
every step it measures is cascade disagreement rather than scene variation.
That is what makes the seam measurement valid rather than merely plausible.

Each threshold is paired with a control that must fail it, so a pass is
evidence the metric has teeth: removing the bias must produce acne, an
exaggerated normal offset must detach the contact shadow, and disabling the
overlap blend must turn the cascade step into a full comparison. Thresholds are
calibrated against the measured values on this scene and recorded in the
[Phase 28 report](../reports/phase-28.md); they are this scene's acceptance
limits, not universal settings.

## Scope and limitations

The quality metrics are CPU analysis tools over the software sampler. They do
not inspect images, do not cover the GPU comparison path, and a passing
threshold on this scene is not a claim about arbitrary scenes: a probe set only
measures what it samples.

Maps and depth buffers remain caller-owned; `ShadowMapView` is a borrowed view.
The CPU atlas is an allocator only: it does not own a texture, render maps, or
integrate with a renderer resource manager. The RHI depth-only render-graph
helper transitions to depth-write, clears reverse-Z depth to zero, and invokes
a caller-owned draw callback; the sampling helper declares graph reads so
hazard tracking orders producers and inserts the required transitions. The
sampler helper configures reverse-Z comparison. Null-backend tests cover graph
execution and descriptor setup. The D3D12 test rasterizes and reads back an
occluder, then schedules a fullscreen diffuse receiver through the graph
sampling pass. Its pixel shader samples shadow visibility per pixel and
combines ambient with shadowed direct illumination; GPU readback verifies
shadowed-versus-lit pixel contrast and the lit out-of-map border. This proves
the GPU comparison-to-lighting path, but is not integration into a production
scene renderer or a full rendered-scene acceptance. The graph helper handles
declared dependencies and transitions, but does not choose, allocate, cull, or
update light maps. There is no production shadowed GPU scene renderer.

The bias defaults are conservative starting values, not universal scene
settings. Projects should tune them against their light projection and depth
range; excessive positive reverse-Z bias can detach shadows. PCF softens
visibility edges but does not estimate a physical emitter's penumbra.

## Research-informed scope boundary

Epic describes Virtual Shadow Maps as 16k virtual maps divided into 128x128
pages, with page allocation/rendering driven by the camera depth buffer and
cached until invalidated; directional lights use clipmaps. Epic's Shadow Map
Ray Tracing (SMRT) then traces multiple light-distributed rays through those
maps for contact-hardening soft shadows. Those are system-level rendering,
culling, cache-invalidation, and GPU-sampling features, not just alternate
filter kernels. This phase implements fixed caller-owned maps, practical
cascades, bounded software PCF/PCSS, and RHI depth-pass/resource helpers. It
does not implement virtual pages, demand-driven GPU rendering, persistent page
caching, SMRT, ray-traced shadows, or renderer-plugin scene integration needed
to approach Unreal Engine's complete shadow pipeline. Consequently, no
Unreal-parity or photorealistic-quality claim is made.

The PCF reference explains why comparison results must be filtered rather than
prefiltering depth itself, and why fixed broad kernels can soften fine detail.
The RHI comparison-sampler helper uses a zero-depth clamp border for reverse-Z:
with the `GreaterEqual` comparison, a receiver outside the map should compare
as lit against the far-depth value. The D3D12 integration test validates
comparison direction, zero-depth border behavior, sampled depth-resource
transitions, and shadowed receiver lighting on the available adapter; it does
not establish image quality in a production scene.

## Verification

The focused tests verify reverse-Z map generation, occlusion, lit samples,
intermediate PCF coverage, PCSS contact hardening, point-face seam blending,
optional soft-filter bounds, depth and normal bias, radius clamping,
out-of-frustum handling, directional/spot/point PBR integration, shadowed fog,
spot/point transform coverage, atlas allocation/reuse, importance resolution
selection, cascade debug colors, cascade frustum coverage, sub-texel
translation stability, all-cascade rendering, all-or-nothing publication,
pixel-identical cascade depth maps for a 0.001-texel camera translation,
visibility blending, render-graph depth transitions,
reverse-Z sampler configuration, D3D12 rasterized-depth readback, and GPU
shadowed receiver output that distinguishes occluded and lit pixels and
validates the zero-depth border. A
software shadow scene is rendered at 15x resolution per axis (225 raster
samples per output pixel), resolved in linear HDR, then tone-mapped; the more
expensive volumetric-fog comparison uses 8x per axis. A synthetic edge test
checks fractional coverage after resolve. This high-quality offline
supersampling is not a real-time anti-aliasing feature or a rendered GPU
acceptance. Software captures compare a
512x512 map at radius 2 with a 2048x2048 map at radius 8; the larger map
substantially reduces the visible floor stippling in this test scene. The
implementation references NVIDIA's
[PCSS integration paper](https://developer.download.nvidia.com/whitepapers/2008/PCSS_Integration.pdf),
Epic's [Virtual Shadow Maps](https://dev.epicgames.com/documentation/en-us/unreal-engine/virtual-shadow-maps-in-unreal-engine)
and [Volumetric Fog](https://dev.epicgames.com/documentation/en-us/unreal-engine/volumetric-fog-in-unreal-engine)
documentation, NVIDIA's [Shadow Map Antialiasing](https://developer.nvidia.com/gpugems/gpugems/part-ii-lighting-and-shadows/chapter-11-shadow-map-antialiasing),
Microsoft's [HLSL SampleCmp reference](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-samplecmp)
and [D3D12 resource-barrier guide](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12),
NVIDIA's [Depth Precision Visualized](https://developer.nvidia.com/content/depth-precision-visualized),
and the [LearnOpenGL shadow-mapping guide](https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping).
The captured four-cascade ground-and-box scene is written as
`phase28_cascaded_shadow_acceptance.ppm` by the software shadow test. Its
camera-motion check proves shadow-map stability for a tiny translation that
does not cross the snapped texel boundary. The acceptance tests add the
measured bar: a four-cascade rendered scene with zero false-shadowed probes
across the shadow distance and on the caster's own light-facing face, an
unbroken contact shadow, a continuous umbra transect across all three splits, a
blended cascade step more than an order of magnitude below the unblended one,
and a 24-frame camera translation whose visibility barely moves. Each is paired
with a control configuration that fails it. These tests do not establish that
arbitrary real scenes are free of acne, detached shadows, cascade seams, or
motion artifacts; they establish it for the probes and the scene they sample.
The detailed
command results and remaining gaps belong in the
[Phase 28 implementation report](../reports/phase-28.md).
