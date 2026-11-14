# 10_antialiasing — the renderer-wide anti-aliasing path

Renders one PBR scene through `cyber/software/antialias.h` at several sample
rates and reconstruction filters, writes a capture for each, and prints how far
each one is from a converged image.

The scene is built from the shapes that alias worst:

- a tiled, normal-mapped floor receding to the horizon (texture minification),
- five spheres sweeping roughness from near-mirror to matte, dielectric on the
  left and metal on the right (specular highlight aliasing),
- a rotated box and a tilted metal torus (hard and curved silhouettes),
- a 17-rod railing about one pixel wide per rod at 960x540 (thin geometry that
  a single-sample rasterizer drops outright),
- three cascaded shadow maps, a point light, a spot light, and height fog.

## How the anti-aliasing works

The software rasterizer has no multisample coverage mask. Instead the whole
frame is rasterized into an `AntiAliasTarget`, which is an N x N supersampled
color and depth pair, and `Resolve` reconstructs the final image with a
windowed filter. Because every sample runs the full pixel stage, geometry
edges, shadow edges, specular highlights, and texture minification are all
anti-aliased by the same mechanism — an edge-only post filter such as FXAA
smooths none of the last three.

Filtering runs in linear light and before tone mapping, so a bright highlight
next to a dark edge reconstructs at its real intensity instead of a clipped
one.

The defaults are the renderer-wide defaults: `AntiAliasQuality::High` (4x4 =
16 samples per pixel) with the Mitchell-Netravali filter.

## Build and run

```
cmake --preset release
cmake --build --preset release
./build/release/samples/10_antialiasing/cyber_sample_antialiasing --out aa_samples
```

With no `--quality` or `--filter`, the sample writes the full comparison set:
`aa_ultra_8x`, `aa_off`, `aa_balanced_2x`, `aa_high_4x`, and one capture per
filter at 4x4. Supplying either option renders that single configuration.

Options: `--out <directory>`, `--width <px>`, `--height <px>`,
`--exposure <stops>`, `--quality off|balanced|high|ultra`,
`--filter box|tent|mitchell|blackman`.

## What it prints

```
capture              samples    filter    seconds  hard steps     vs 8x
aa_ultra_8x               64  mitchell      49.88      2.081% reference
aa_off                     1       box       1.06      3.187%     12.00
aa_balanced_2x             4  mitchell       3.73      2.176%      3.61
aa_high_4x                16  mitchell      13.75      2.096%      1.58
filter_box_4x             16       box      14.25      2.424%      4.24
filter_tent_4x            16      tent      13.77      2.003%      1.73
filter_mitchell_4x        16  mitchell      13.93      2.096%      1.58
filter_blackman_4x        16  blackman      13.89      2.014%      1.80
```

- `hard steps` is the fraction of neighbouring pixel pairs whose sRGB luminance
  jumps by 48/255 or more. A staircase is made of hard steps, so the number
  falls as anti-aliasing improves.
- `vs 8x` is the root-mean-square 8-bit sRGB channel difference against the
  8x8 capture, which stands in for ground truth. It is the number that matters:
  one sample per pixel sits at 12.0, and 4x4 Mitchell closes that to 1.6.
  Unlike an edge count it also catches thin geometry that a low sample rate
  drops entirely.

At a fixed 4x4, Mitchell is closest to the converged image and box is furthest,
which is what the wider reconstruction support buys.

## Cost

Supersampling costs the full pixel stage per sample, so time scales with the
sample count: roughly 1 s at 1x, 3.7 s at 2x2, 14 s at 4x4, and 50 s at 8x8 for
this scene at 960x540 on a 4-core CPU. Shadow-map rendering is per frame, not
per sample, so it does not scale with the rate.

## Known limitation

The CPU pixel stage computes no screen-space derivatives, so there is no
per-pixel texture LOD; the floor uses a fixed mip level. Supersampling is what
keeps the distant tiles from shimmering. Derivative-driven LOD would reduce the
sample rate needed for the same result.
