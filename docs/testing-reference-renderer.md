# Phase 26 reference renderer

Status: complete. The deterministic CPU path tracer is independent of the
production BRDF and pixel shader.

## Current comparison contract

`cyber_sample_lighting` renders the same 7x7 metallic-roughness sphere layout
in two ways:

1. The forward image uses tessellated spheres, the software triangle
   rasterizer, perspective-correct varyings, the Cook-Torrance pixel shader,
   HDR accumulation, and ACES tone mapping.
2. The reference image uses analytic ray-sphere intersections, a separately
   implemented Cook-Torrance BRDF, deterministic cosine/GGX path sampling,
   shadow rays for directional/point/spot lights, and environment bounces. It
   shares scene/material/light inputs, but not the production BRDF or pixel
   shader.

The scene exercises the same procedural linear base-color texture and tangent-
space normal map in both renderers. Both outputs receive the same ACES exposure
transform and are compared in 8-bit sRGB RGB space. The reported mean absolute
error is

```text
sum(abs(forward_srgb8 - reference_srgb8)) / (pixel_count * 3 * 255)
```

Coverage mismatch is the fraction of pixels whose forward/reference alpha
coverage masks differ. Run a capture from the repository root:

```text
cmake --build build/phase26-debug --target cyber_sample_lighting
build/phase26-debug/samples/04_lighting/cyber_sample_lighting.exe --capture build/phase26-forward.ppm --reference build/phase26-reference.ppm
```

The sample prints measured values and exits unsuccessfully if the mean error
exceeds `0.015` or coverage mismatch exceeds `0.1%`. These bounds provide
approximately 1.85x and 2.38x headroom over the largest observed default/
64-sample measurements, respectively.

## Acceptance evidence

On Windows 10 22H2 with an Intel Core i5-6400 and MSVC 19.44, the Debug
`cyber_sample_lighting` capture completed at native, 4x, and 8x supersampled
resolutions. All runs use four maximum bounces and seed `0xC0FFEE26`. The
default 8x run distributes 64 path samples over each output pixel's 8x8
subpixel grid, then box-filters both images to 480x360.

| Supersampling | Path samples/output pixel | Runtime | Mean sRGB absolute error | Coverage mismatch |
|---:|---:|---:|---:|---:|
| 8x (default) | 64 | 80,474 ms | 0.00879 | 0.030% |
| 4x | 16 | 27,268 ms | 0.00864 | 0.035% |
| 1x | 16 | 19,901 ms | 0.00807 | 0.000% |
| 1x | 64 | 57,958 ms | 0.00809 | 0.000% |

At 8x supersampling, each output pixel resolves an 8x8 grid of samples; the 64
path samples/output pixel are spread over that grid. The native-resolution
convergence run confirms the metric is stable as path samples increase.
Supersampling further improves the smoothness of geometric edges. The
remaining color error reflects rasterization-versus-ray shading and
interpolation differences. The limits allow numerical headroom for other
supported builds while remaining measurably tighter than the observed error
envelope. The `software_pbr` white-furnace,
BRDF, normal-map, linear base-color texture, and light tests pass; the separate
`lighting_reference` tests verify determinism, lit output, background alpha,
and invalid-configuration rejection.

Commands:

```text
cmake --build build/phase26-debug --target cyber_test_software_pbr cyber_sample_lighting
ctest --test-dir build/phase26-debug -R "software_|lighting_reference" --output-on-failure
build/phase26-debug/samples/04_lighting/cyber_sample_lighting.exe --capture build/phase26-forward.ppm --reference build/phase26-reference.ppm
build/phase26-debug/samples/04_lighting/cyber_sample_lighting.exe --capture build/phase26-forward-aa-8x.ppm --reference build/phase26-reference-aa-8x.ppm --supersample 8 --samples 64
build/phase26-debug/samples/04_lighting/cyber_sample_lighting.exe --capture build/phase26-forward-native.ppm --reference build/phase26-reference-native.ppm --supersample 1
build/phase26-debug/samples/04_lighting/cyber_sample_lighting.exe --capture build/phase26-forward-64.ppm --samples 64 --supersample 1
```

The path tracer must not silently fall back to the production pixel shader:
failure to allocate a target or to complete a reference render fails the
sample and its validation command. Full hardware/compiler details and the
commands used for acceptance are recorded in the
[Phase 26 report](reports/phase-26.md).
