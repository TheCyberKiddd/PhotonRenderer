# cyber_software_renderer

**Status:** active · **Layer:** 4 · **Kind:** product

CPU reference renderer: framebuffer, rasterization, depth, textures, shaders,
golden images, and the second plugin.

The binding contract for this module lives in [docs/MODULE.md](docs/MODULE.md).
Read it before adding code or changing a public header.

## Layout

- `include/cyber/software/` — public headers, self-contained.
- `src/` — private implementation.
- `tests/` — unit tests owned by this module.

## Phase 6: framebuffer and present

- Linear `RGBA8Unorm` and `RGBA32F` framebuffers with explicit stride,
  allocator-backed storage, clear/fill/pixel/blit/convert.
- `PackPresentPixels` converts linear pixels to sRGB RGBA8 for the platform
  presenter; `WriteImage` exports PPM or TGA deterministically.
- `FramePacer` schedules 60 Hz deadlines on the monotonic clock and reports
  frame statistics for the overlay.
- A 5x7 bitmap font draws the frame-time readout.

```cpp
cyber::software::FramebufferDesc desc{};
desc.width = 640;
desc.height = 360;
desc.format = cyber::software::PixelFormat::RGBA32F;
auto created = cyber::software::Framebuffer::Create(desc, heap);
created.Value().Clear(cyber::math::Color{0.1f, 0.2f, 0.3f});
```

## Phase 7: triangle rasterization

- `DrawLine` clips a segment to the framebuffer and walks it with integer
  Bresenham.
- `FillTriangle` fills one triangle with the exact top-left rule and linear
  barycentric color.
- `RasterizeTriangles` bins triangles into screen tiles, runs one job per tile,
  and produces bytes independent of worker count and tile size.
- Positions snap to 24.8 fixed point. Shared vertices must be supplied with the
  same value; the coverage test builds its mesh that way.

```cpp
const cyber::software::RasterVertex a{cyber::math::Vec2{10.0f, 10.0f},
                                      cyber::math::Color{1.0f, 0.2f, 0.2f, 1.0f}};
const cyber::software::RasterVertex b{cyber::math::Vec2{90.0f, 10.0f},
                                      cyber::math::Color{0.2f, 1.0f, 0.2f, 1.0f}};
const cyber::software::RasterVertex c{cyber::math::Vec2{10.0f, 90.0f},
                                      cyber::math::Color{0.2f, 0.2f, 1.0f, 1.0f}};
const cyber::software::RasterTriangle triangle{a, b, c};
cyber::software::RasterStats stats{};
auto status = cyber::software::RasterizeTriangles(
    framebuffer, cyber::memory::Span<const cyber::software::RasterTriangle>(&triangle, 1),
    &jobs, heap, {}, &stats);
```

## Phase 8: depth, clipping, and perspective

- `DepthBuffer` is a float reverse-Z target: cleared to 0, greater-or-equal
  test, near plane at 1.
- `DrawTriangles` runs the vertex pipeline: `projection * view * model`, a
  homogeneous near-plane clip, the perspective divide, the viewport y flip,
  back-face culling, and the +/-16384-pixel guard band.
- Color is perspective-correct (`attribute / w` and `1 / w`); depth interpolates
  linearly because NDC depth is affine in screen space.
- `Camera` plus `FlyCamera` and `OrbitCamera` controllers take plain float
  deltas, so they are testable without a window and the sample owns the input
  mapping.
- `ClipTriangleNearPlane` is public so the clipper can be tested directly; it
  returns 3 or 4 vertices in input order.

```cpp
cyber::software::Camera camera;
camera.position = cyber::math::Vec3{0.0f, 1.0f, 5.0f};
camera.pitchRadians = -0.15f;

cyber::software::PipelineDesc desc{};
desc.viewport.width = 1280.0f;
desc.viewport.height = 720.0f;
desc.view = cyber::software::CameraView(camera);
desc.projection = cyber::software::CameraProjection(camera, 1280.0f / 720.0f);
desc.cullMode = cyber::software::CullMode::Back;

color.Clear(cyber::math::Color{0.05f, 0.07f, 0.10f});
depth.Clear(0.0f);
const auto status = cyber::software::DrawTriangles(
    desc, cyber::memory::Span<const cyber::software::Vertex>(vertices), color, depth,
    &jobs, heap, &stats);
```

`samples/03_depth` renders the rotating cube and the perspective floor; its
`--inside` capture places the camera at the cube centre so the near clipper is
visible in the reference image.

## Phase 27: clustered PBR reference

`clustered_pbr.h` adapts world-space point lights to the renderer's
view-space cluster bounds and consumes the resulting cluster indices in the
shaded raster path. `ClusteredPbrPixelShader` uses the full local-light list
when the pixel is outside the grid or a list does not match the light span,
preserving image correctness. This is the CPU reference integration; the
GPU-built lists, GPU shading consumer, and rendered-frame benchmark remain
outstanding.

## Phase 30: post-processing and anti-aliasing

`post_process.h` owns `PostProcessStack`, the HDR chain that runs between
shading and display: ambient occlusion, depth of field, motion blur, bloom,
exposure, tone mapping, and colour grading, in that fixed physical order.
Every pass toggles at runtime and reports its own CPU time; a pass that is off
or whose input buffer is absent is skipped rather than failed, and still
appears in the timing table.

`temporal_aa.h` owns the anti-aliasing paths: Halton camera jitter,
`TemporalAaResolver` for velocity-reprojected accumulation with YCoCg
neighbourhood rectification, flicker-relaxed box widening, and depth-based
disocclusion rejection, plus `FxaaResolve` as the cheap alternative and the
no-history fallback. `AntiAliasMethod::Supersample` selects the renderer-wide
path in `antialias.h`; MSAA does not exist here, because a software rasterizer
has no fixed-function coverage mask to resolve.

`post_process_quality.h` measures the result: image difference, ghosting,
temporal stability, and edge aliasing, so the phase's quality bar is a number
with a threshold rather than an inspection.

```cpp
auto stack = cyber::software::PostProcessStack::Create(width, height, heap);

cyber::software::PostProcessInputs inputs{};
inputs.color = &hdrColor;
inputs.depth = &depth;
inputs.normals = &normals;
inputs.velocity = &velocity;

cyber::software::PostProcessSettings settings{};
settings.enableDepthOfField = true;

cyber::software::PostProcessTimings timings{};
const auto status = stack.Value().Execute(inputs, display, settings, view,
                                          deltaSeconds, &jobs, &timings);
```

Both paths write identical bytes with and without a job system, which is what
lets the acceptance suite assert on exact numbers.

## Phase 31: global illumination and ray tracing

`raytrace.h` is a CPU ray tracer in DirectX Raytracing's shape: a
`BottomLevelAccelerationStructure` over object-space triangles, a
`TopLevelAccelerationStructure` over transformed, masked instances, and
closest-hit and any-hit queries described by a `RayDesc`. Both levels are
built with a binned surface-area heuristic, and the depth bound lets every
query run on a fixed stack.

`global_illumination.h` uses it for light transport. `TracePathRadiance` is a
Lambertian path tracer with next-event estimation and Russian roulette;
`IrradianceVolume` is a grid of order-2 SH probes, each with a 16x16 octahedral
map of visibility-distance moments, baked by `BakeIrradianceVolume` and read by
`SampleIrradianceVolume` with normal and Chebyshev visibility weighting. Set
`PbrLightingData::irradianceVolume` and the forward shader uses the bake as its
diffuse ambient term. [ADR 0028](../../docs/adr/0028-gi-approach.md) records
why baked volumes ship first.

`denoise.h` holds `SpatioTemporalDenoiser`, an SVGF-style filter for
one-sample ray-traced signals, and `gi_quality.h` measures accuracy, light
leakage, and residual noise so the phase's bar is a number.

```cpp
cyber::software::GiScene scene{};
scene.geometry = &topLevel;
scene.materials = materials;
scene.pointLights = lights;

auto volume = cyber::software::IrradianceVolume::Create(volumeDesc, heap);
cyber::software::IrradianceBakeDesc bake{};
bake.raysPerProbe = 1024;
const auto status = cyber::software::BakeIrradianceVolume(scene, bake, volume.Value(),
                                                          nullptr, &jobs);

cyber::software::IrradianceVolumeView view{};
view.volume = &volume.Value();
lighting.irradianceVolume = &view;  // PbrLightingData
```

The bake and the denoiser write identical bytes with and without a job system.
GPU ray tracing is not part of this module; `cyber::renderer::SelectRayTracingPlan`
decides which technique answers each effect from what the device reports.

## Roadmap

Phases 6, 7, 8, 9, 25, 26, 27, 28, 29, 30, 31.
