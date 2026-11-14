# Phase 26 — implementation summary

Status: complete.

Phase 26 delivers a CPU forward Cook-Torrance metallic-roughness shader using
GGX distribution, Smith-GGX visibility, Schlick Fresnel, and Lambert diffuse;
tangent-space normal mapping; directional, inverse-square point, and spot
lights; and ACES HDR tone mapping. Base-color texture samples are linearized
from sRGB when authored, while normal-map data remains linear. `software_pbr`
covers BRDF behavior, a deterministic white-furnace integration, tangent
handedness, normal-map shading, linear texture sampling, lighting, and tone
mapping.

Sample 04 renders a normal-mapped, textured 7x7 sphere grid and compares the
forward rasterizer against an independent deterministic CPU path tracer. On
Windows 10 22H2 (Intel Core i5-6400, MSVC 19.44), the default 8x
supersampled, 64-path-sample run measured `0.00879` mean sRGB absolute error
and `0.030%` coverage mismatch. The enforced limits are `0.015` mean error and
`0.1%` coverage mismatch.

Details:

- [Design and equations](../design/phase-26-pbr.md)
- [Reference renderer contract and acceptance evidence](../testing-reference-renderer.md)
- [Validation report](../reports/phase-26.md)
- [Runnable sphere-grid sample](../../samples/04_lighting/README.md)
