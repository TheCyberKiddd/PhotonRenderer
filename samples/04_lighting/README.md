# Sample 04 — physically based lighting

Builds a 7x7 metallic-roughness sphere grid with the CPU forward rasterizer.
Columns increase metalness; rows increase perceptual roughness. The scene
combines a directional light, an inverse-square point light, a spot light,
diffuse ambient irradiance, and ACES tone mapping. By default, the 480x360
output is rendered at 8x resolution per axis (3840x2880) and box-filtered down
to smooth geometric edges. The reference uses 64 path samples per output pixel
and four bounces.

```text
cyber_sample_lighting --capture build/phase26-forward.ppm \
  --reference build/phase26-reference.ppm --supersample 8 --samples 64
```

For maximum offline edge smoothness, use `--supersample 15 --samples 225`.

Use `--supersample 1` through `15` for progressively smoother output. This is
the linear scale per axis: 15x renders 225 subpixel samples for every output
pixel before the box-filter resolve. Resolve averages linear HDR values before
tone mapping. The path tracer distributes its sample budget across that grid;
`--samples` must be a positive multiple of the grid area. For example,
`--supersample 15 --samples 225` uses one path sample at each of the 225
subpixel locations. High factors significantly increase render-target memory
and render time.

The optional reference image uses analytic sphere intersections and an
independently implemented deterministic CPU path tracer with separate BRDF
equations, direct-light shadow rays, and environment bounces. Both images use
the same linear base-color texture and tangent-space normal map. The reported
metric is mean absolute RGB error in sRGB space and pixel coverage mismatch;
the sample exits unsuccessfully if either documented acceptance limit is
exceeded. The default reference uses 16 samples/pixel and four bounces; pass
`--samples 64` for the recorded convergence run. See the
[Phase 26 reference contract](../../docs/testing-reference-renderer.md) and
[validation report](../../docs/reports/phase-26.md). The white-furnace
numerical test is `software_pbr_white_furnace_does_not_gain_energy`.

No third-party rendering library is used. This example links the repository's
own CPU software renderer, platform, memory, and job modules.
