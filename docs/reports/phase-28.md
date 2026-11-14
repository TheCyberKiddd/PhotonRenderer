# Phase 28 implementation report — shadows

Status: software scope complete. Software PCSS, shadowed height fog, cascades,
CPU allocation helpers, depth-only shadow-map rendering, tested D3D12
depth-map rendering, and a measured visual-acceptance suite for acne, detached
shadows, cascade seams, and camera-motion shimmer are implemented. GPU
production-scene integration is outside this phase's scope and remains
unbuilt.
Date: 2026-11-09
Design: [Phase 28 shadows](../design/phase-28-shadows.md)

## Delivered

- Added a non-owning `ShadowMapView` over the software renderer's reverse-Z
  `DepthBuffer`, with caller-provided clip transforms and bounded PCF.
- Added `RenderDirectionalShadowMaps`, which validates and renders an entire
  directional cascade set, attaches the borrowed depth views only after every
  cascade succeeds, and reports per-cascade raster statistics.
- Added constant, slope-scaled, and world-space normal-offset bias. Visibility
  affects direct light only; ambient illumination remains unchanged.
- Added shadow-map rendering using the clipped reverse-Z raster pipeline. This
  first shipped with a caller-owned scratch color target and was later
  replaced by the depth-only path described below.
- Added four practical directional cascades with overlap blending, rounded
  frustum bounds, padded light depth, and texel-snapped light-space translation.
  Cascade selection is integrated with directional PBR lighting.
- Added spot and six-face point-light reverse-Z projection builders, seam-
  blended point-light PBR sampling, and caller-owned shadow maps.
- Added a fixed-tile CPU atlas allocator with guarded power-of-two allocations,
  generation-checked release/reuse, and importance-based resolution selection.
- Added optional deterministic stratified 16-tap PCF and a directional cascade
  visualization toggle for the software PBR path. Soft PCF supports a bounded
  16-texel radius while remaining at 16 comparisons per receiver; hard PCF
  remains bounded to 81 comparisons.
- Added contact-hardening PCSS with bounded blocker search, reverse-Z distance
  linearization, emitter-size penumbra estimation, and bounded filtering.
- Added point-shadow cube-face seam blending and corrected reverse-Z comparison
  sampler selection to `GreaterEqual`.
- Added default-off software height fog with height-dependent extinction,
  Henyey-Greenstein scattering, and shadow lookups for directional, point, and
  spot lights along the view ray.
- Added a render-graph depth-only pass helper that declares depth-write access,
  clears reverse-Z depth to zero, and records a caller-provided draw callback.
  A multi-map sampling pass declares shader-read accesses so the graph orders
  producers and inserts resource transitions. Also added a reverse-Z
  comparison-sampler descriptor helper and GPU shadow-depth resource lifecycle
  helpers.
- Added depth-only graphics pipeline support to the RHI, enabling real
  depth-only rendering without a dummy pixel shader or color target.
- Added a Null-backend test for render-graph execution, two-map producer /
  consumer scheduling, shader-read transition, and comparison-sampler
  configuration. The D3D12 integration test renders an occluder into a depth
  map, verifies its center depth by GPU readback, then runs a fullscreen
  receiver-lighting pass through `AddShadowSamplingPass`. Its pixel shader
  combines ambient and shadowed direct diffuse light using per-pixel
  comparison-sampler visibility. GPU readback verifies that a clear receiver
  is substantially brighter than an occluded receiver and that out-of-map
  comparison is lit.
- Added tests for reverse-Z occlusion, lit samples, fractional PCF, soft-filter
  bounds, bias and out-of-frustum behavior, directional/spot/point PBR
  integration, local-light transforms, atlas allocation/reuse, resolution
  selection, cascade debug colors, cascade fitting/stability/blending, and
  rendered shadow depth, PCSS contact hardening, point-face seam blending,
  and fog attenuation/scattering under directional shadows.
- Added an end-to-end four-cascade ground-and-box capture and a camera-motion
  regression that renders all cascades before and after a 0.001-texel camera
  translation and checks every depth texel for equality. Invalid partial
  target sets are rejected before map handles are published.
- Added a software ground-and-box acceptance scene rendered at 15x resolution
  per axis (3840x2880; 225 samples per output pixel). It resolves linear HDR
  before tone mapping to 256x192, and compares 512x512 and 2048x2048 shadow
  maps; its more expensive volumetric-fog comparison uses 8x per axis.
  The 2048 map removes most of the structured floor stippling visible with the
  512 map in this scene.
- Added an edge-coverage regression test for the 15x supersampling resolve and
  extended the Phase 26 lighting capture tool to allow `--supersample 1-15`.
- Replaced the shadow-map draw with a real depth-only path. `raster.h` gained
  `DepthOnlyRasterVertex`/`DepthOnlyRasterTriangle` and
  `RasterizeDepthOnlyTriangles`, and `shader.h` gained `DepthOnlyShaderDesc`,
  `TransformDepthOnlyVertexShader`, and `DrawDepthOnlyTriangles`. Neither
  carries varyings through the near-plane clipper, binds a color target, nor
  calls a pixel shader. `RenderShadowMap` and `RenderDirectionalShadowMaps` no
  longer take a scratch color target, which removes the documented
  scratch-target limitation and the per-cascade color allocation.
- Added `shadow_quality.h`: `MeasureShadowAcne`, `MeasureShadowContact`,
  `MeasureShadowTransect`, and `MeasureShadowStability`, with the metric
  structs each returns. They drive caller-supplied `ShadowProbe` sets through
  `EvaluateDirectionalShadowVisibility`, so they measure the shipped sampler;
  they allocate nothing, are deterministic, and reject empty, unordered,
  mis-sized, and non-finite input.
- Added the Phase 28 acceptance test suite over a rendered four-cascade
  ground-and-wall scene, covering the roadmap's acne, detached-shadow, and
  cascade-seam bar plus camera-motion shimmer. Every threshold is paired with a
  control configuration that must fail it.
- Added depth-only parity coverage: the depth-only rasterizer's coverage,
  stats, and depth texels match the color depth-tested path exactly and are
  unchanged by job scheduling and tile size, and `DrawDepthOnlyTriangles`
  matches `DrawShadedTriangles` stat for stat and texel for texel on geometry
  that crosses the near plane.

## Validation

All commands were run in the Visual Studio 2022 Build Tools developer
environment on the Intel HD Graphics 530 host.

- `cmake --build build\phase27-debug -j 8` then
  `ctest --test-dir build\phase27-debug --output-on-failure` — the full Debug
  suite passed, 131/131 in 169.89 seconds. This includes the D3D12 shadow
  depth rendering and graph-scheduled shader-read transition, the four-cascade
  software capture, the sub-texel depth-map comparison, the new acceptance
  suite, and the depth-only parity tests.
- `cmake --build build\phase27-release -j 8` then
  `ctest --test-dir build\phase27-release --output-on-failure` — the full
  Release suite passed, 131/131 in 62.25 seconds. This closes the gap the
  previous revision of this report recorded, where Release had not been rerun.
- `ctest --test-dir build\phase27-debug -R software_shadow_acceptance --output-on-failure`
  — passed, 1/1 in 0.70 seconds. The five acceptance tests run in 83, 71, 77,
  415, and 0.03 milliseconds; the metrics they report are in the table above.
- `ctest --test-dir build\phase27-debug -R renderer_shadow_pass --output-on-failure` — passed, 1/1, including D3D12 depth rendering, graph-scheduled shader-read transition, shaded pixel contrast, and out-of-map comparison.
- The depth-only change made the software shadow test faster, not slower: the
  same test that took 143.64 seconds with the scratch-color path now takes
  116.50 seconds, with byte-identical cascade depth maps.
- `py scripts\check_structure.py` — passed, 17 modules, 1 contract.
- `py scripts\layer_check.py` — passed, 17 modules.
- `git diff --check` — passed.
- `clang-format` is still unavailable in this environment, so
  `scripts\check_format.py` could not run; the new code follows the
  surrounding style by hand.

The 15x software captures show antialiased edges; the low-poly test scene is
not evidence of ultra-realistic shadow quality. The GPU sample test verifies
the real GPU shadow-comparison-to-lighting path for a simple diffuse receiver,
not a production GPU PBR renderer or complex scene integration. The
four-cascade software capture was inspected and shows shadow contribution on
all three scene objects. The acceptance suite replaces the depth-map sweep's
single sub-texel step with a 24-frame camera translation, but it measures
probe visibility rather than comparing rendered image sequences. Generated PPM
captures remain in the ignored build directory and are not checked into the
report.

## Measured acceptance

Thresholds are calibrated against these measurements on the acceptance scene
and are not universal settings. Each row's control is the configuration that
must fail the same metric, which is what makes the threshold evidence rather
than an assertion that happens to hold.

| Artifact | Measured | Threshold | Control (must fail) |
|---|---|---|---|
| Acne | 0 of 1,520 probes darkened; min and mean visibility 1.000000 | 0 darkened, min within 1e-6 of 1 | All biases zero: 783 of 1,520 darkened (51.5%), min 0.000000 |
| Detached shadow | contact visibility 0.000000; no lit sample over 1.4 world units | contact below 0.02, no lit sample | `normalOffsetBias = 1.4`: contact visibility 1.000000, gap 0.0000 |
| Cascade seam (umbra) | maxStep 0.000000 over 3 split crossings | maxStep and at-split step below 0.01 | — |
| Cascade seam (worst offset) | maxStep 0.102138, at-split 0.030643, mean 0.001391 | at-split below 0.1, maxStep below 0.15 | `blendFraction = 0`: maxStep and at-split 1.000000 |
| Shimmer (umbra, lit ground, caster face) | maxFrameDelta 0.000000, max total variation 0.000000 over 24 frames | maxFrameDelta below 0.01, total variation below 0.05 | — |
| Shimmer (shadow edge) | maxFrameDelta 0.004507, mean 0.000061, max total variation 0.016972 | maxFrameDelta below 0.05, total variation below 0.1 | — |

The seam offset is the measured worst case for this scene: the coarse far
cascade quantizes the wall's shadow edge one texel short of the near cascade's,
so a probe there is shadowed in one cascade and lit in the next. The overlap
blend reduces that full-comparison disagreement by roughly 32x at the split.
The transect is sampled every 0.05 world units of depth so the blend's ramp is
resolved as many small steps; a coarser transect would report the ramp itself
as a large step and could not distinguish a smooth blend from a hard seam.

## Not yet delivered

The render graph schedules depth producers and multi-map shader readers, and
the D3D12 test rasterizes a shadow map, transitions it to shader-read, renders
a fullscreen receiver with shadowed direct lighting, and verifies the shaded
pixel contrast by readback. This validates the GPU shadowed-lighting
mechanism, but is a focused RHI test, not integration into a production GPU
scene renderer; the repository still does not provide that renderer, and the
acceptance measurements cover the software sampler only. Point faces and the
CPU atlas still require caller-owned depth and storage. The acceptance scene is
a ground plane and a wall measured through probe sets, not a sweep over complex
real scenes or an image-sequence comparison, so the result is freedom from
these artifacts for what it samples, not a general guarantee. No Unreal parity
is claimed: virtual shadow pages, demand-driven GPU rendering, persistent page
caching, SMRT, and ray-traced shadows remain unimplemented.

## Research references

- NVIDIA, [PCSS Integration](https://developer.download.nvidia.com/whitepapers/2008/PCSS_Integration.pdf).
- Epic Games, [Virtual Shadow Maps](https://dev.epicgames.com/documentation/en-us/unreal-engine/virtual-shadow-maps-in-unreal-engine) and [Volumetric Fog](https://dev.epicgames.com/documentation/en-us/unreal-engine/volumetric-fog-in-unreal-engine).
- LearnOpenGL, [Shadow Mapping](https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping).
