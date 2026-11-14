# Phase 26 validation report

Status: complete.

## Delivered

- CPU metallic-roughness forward shading with GGX/Smith/Schlick, directional,
  point, and spot lights, linear HDR accumulation, and ACES output mapping.
- Tangent-space normal mapping and base-color texture sampling in linear space.
- An independent deterministic analytic-sphere path tracer with its own BRDF
  evaluation, cosine/GGX sampling, shadow rays, and environment bounces.
- White-furnace, BRDF, light, texture, normal-map, determinism, output, and
  invalid-configuration tests.
- A normal-mapped 7x7 sphere-grid sample that fails when image error or
  coverage mismatch exceeds its acceptance limits.

## Reference comparison

Environment: Windows 10 22H2, Intel Core i5-6400, MSVC 19.44, Debug build in
`build\phase26-debug`. The path tracer used seed `0xC0FFEE26` and four maximum
bounces. Timings are wall-clock measurements and may vary with system load.

| Supersampling | Path samples/output pixel | Runtime | Mean sRGB absolute error | Coverage mismatch | Result |
|---:|---:|---:|---:|---:|:---|
| 8x (default) | 64 | 80,474 ms | 0.00879 | 0.030% | Pass |
| 4x | 16 | 27,268 ms | 0.00864 | 0.035% | Pass |
| 1x | 16 | 19,901 ms | 0.00807 | 0.000% | Pass |
| 1x | 64 | 57,958 ms | 0.00809 | 0.000% | Pass |

At 8x supersampling, each output pixel resolves an 8x8 grid of samples; the 64
path samples/output pixel are spread over that grid. At native resolution, 64
samples produced the same coverage as the 16-sample run and nearly identical
mean error, confirming convergence. The accepted limits are `0.015` mean sRGB
absolute error and `0.1%` coverage mismatch, approximately 1.74x and 2.86x the
largest measurements after antialiasing.

## Validation commands

The build command was run in a Visual Studio Developer Command Prompt
environment to select the configured MSVC toolchain:

```text
cmake --build build\phase26-debug --target cyber_sample_lighting cyber_test_software_pbr cyber_test_lighting_reference --parallel 8
```

Result: passed. The full relevant software-renderer and path-tracer test
selection passed 21/21:

```text
ctest --test-dir build\phase26-debug -R "software_|lighting_reference" --output-on-failure
```

Both acceptance captures passed:

```text
build\phase26-debug\samples\04_lighting\cyber_sample_lighting.exe --capture build\phase26-forward-aa-8x.ppm --reference build\phase26-reference-aa-8x.ppm --supersample 8 --samples 64
build\phase26-debug\samples\04_lighting\cyber_sample_lighting.exe --capture build\phase26-forward-64.ppm --samples 64 --supersample 1
```

Repository structure and layer checks passed:

```text
py -3 scripts\check_structure.py
py -3 scripts\layer_check.py
```

`git diff --check` and clang-format verification of all changed C++ files
passed. The repository-wide format check reports an existing unrelated
formatting violation at `engines\cyber_scene\src\mesh.cpp:757`; that file was
not changed as part of Phase 26.

## Reproduction

See the [reference renderer contract](../testing-reference-renderer.md) for
the image metric, accepted limits, and capture commands. The default 8x
supersampling test verifies the acceptance gate.
