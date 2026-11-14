# Photon Renderer roadmap status

Snapshot: 2026-11-14

This is a factual project-state snapshot to help prepare a revised product
roadmap. The phase names and intended outcomes follow
[`graphics/CYBERGRAPHICS_ROADMAP.md`](graphics/CYBERGRAPHICS_ROADMAP.md);
this document does not replace that roadmap or change any phase checkbox.

## Executive status

- **Phases 1-14:** marked complete in [`ROADMAP.md`](ROADMAP.md), with
  implementation and evidence described in the existing phase reports.
- **Phase 15:** complete. The Intel HD Graphics 530 D3D12 sample-path bug is
  fixed, the sample submits and presents, and the fresh RenderDoc/PIX captures
  confirm the nested `Clear Pass` and `Clear backbuffer` labels and the blue
  presented frame. The raw native `SetMarker` support API remains unbundled and
  is validated through the RHI tests rather than a new external dependency.
- **Phase 16:** complete. The compiler emits HLSL used by the D3D12 cube
  comparison; Debug, Release, and Debug-ASan each pass 88/88, the package,
  structure, and layer checks pass, and the repository-wide formatter passes.
- **Phase 17:** complete. Compiler-owned
  reflection, permutations, disk cache, asynchronous compilation and pipeline
  swaps, and the D3D12 live-reload sample are implemented. Debug, Profile,
  Release, Shipping, and Debug-ASan each pass 96/96; package, structure, layer,
  formatting, and live-reload checks pass. The valid edit, syntax-error
  rejection, and restore measured 48 ms, 49 ms, and 84 ms on this PC. Its
  implementation and evidence are included in the Phase 17 commit.
- **Phase 18:** implementation and local validation complete; Debug, Release,
  and Debug-ASan each pass 98/98. The D3D12 sample self-test verifies 100
  material instances, sorted draws, and a live parameter change at frame 20.
  The package and source consumers and all structure, layer, formatting, and
  whitespace checks pass. Implementation and evidence are included in this
  commit, and the master roadmap marks the phase complete. See the
  [Phase 18 report](reports/phase-18.md).
- **Phase 19:** complete. On the Intel HD Graphics 530, the Release bindless
  path submits 10,000 unique-transform draws with p95 completion of 11.6 ms,
  under the 16.67 ms budget. Debug, Release, and Debug-ASan pass 99/99;
  package/source consumers and structure, layer, formatting, and whitespace
  checks pass. The slower dynamic-offset comparison and explicit WARP results
  are retained in the [Phase 19 report](reports/phase-19.md).
- **Phase 20:** complete. The render graph infers dependencies and barriers,
  culls unused passes, aliases compatible non-overlapping transients, and
  writes bounded text and DOT diagnostics. Debug, Release, and Debug-ASan pass
  100/100; package/source consumers and structure, layer, formatting, and
  whitespace checks pass. See the [Phase 20 report](reports/phase-20.md).
- **Phase 21:** complete. Debug, Release, and Debug-ASan pass 101/101; the Linux
  ThreadSanitizer render-thread and hot-pipeline targets pass; package/source
  consumers and repository checks pass. See the
  [Phase 21 report](reports/phase-21.md).
- **Phase 22:** complete; scene proxies and culling are documented in the
  Phase 22 report.
- **Phase 23:** complete; geometry processing and indirect batches are
  documented in the Phase 23 report.
- **Phase 24:** implementation and local validation complete. The Phase 24
  report records parser/cooker tests, fuzz mutations, benchmark, and known
  limits.
- **Phase 25:** complete. The versioned C ABI, software and D3D12 renderer
  plugins, borrowed native-window surface, capability negotiation, startup
  preference, and fallback after device-creation failure are implemented.
  Windows validation includes a hidden-window D3D12 triangle submission,
  callback reporting, software output, legacy ABI compatibility, and fallback
  tests. Renderer preference is exposed at the host API; no engine executable
  or persisted settings UI exists in this repository. See the
  [Phase 25 design note](design/phase-25-engine-interface.md) and
  [plugin ABI](plugin-abi.md).
- **Phase 26:** complete. The forward PBR renderer, normal-mapped sphere grid,
  independent deterministic CPU path tracer, white-furnace and renderer tests,
  and enforced image thresholds are implemented. The 16- and 64-sample
  8x-supersampled comparison measures `0.00879` mean sRGB error and `0.030%`
  coverage mismatch. See the [Phase 26 summary](graphics/PHASE_26.md) and
  [validation report](reports/phase-26.md).
- **Phase 27:** the CPU clustered-forward path is complete, with a standalone
  D3D12 compute proof validated on the current adapter. The software renderer
  builds and consumes CPU cluster lists, checks rendered 1,000-light PBR
  output against brute force, exposes heat-map occupancy data, and includes an
  end-to-end CPU frame benchmark. The D3D12 sample validates bounded GPU
  cluster lists and representative synthetic-surface PBR output against CPU
  references. Its scope is not a rasterized scene, renderer-plugin integration,
  automatic backend switching, a visual heat-map overlay, or a deferred
  comparison. See the [Phase 27 report](reports/phase-27.md).
- **Phase 28:** software scope complete. The software renderer adds
  contact-hardening PCSS, point-cube seam blending, and default-off height fog
  with shadowed directional/point/spot scattering, alongside reverse-Z maps,
  cascades, and PCF. Shadow maps now render through a real depth-only raster
  path that binds no color target and runs no pixel shader, so the previous
  scratch-color-target requirement is gone; its coverage, stats, and depth
  texels match the color path exactly. The software path has an end-to-end
  directional cascade renderer with atomic map publication and a four-cascade
  scene capture, and a sub-texel camera-motion regression verifies
  pixel-identical cascade maps. The RHI owns GPU shadow-depth resource helpers,
  depth-only graphics pipelines, and graph scheduling from multiple depth-map
  producers to a shader-reading consumer. A D3D12 render-graph test rasterizes
  geometry, reads back depth, and validates reverse-Z comparison samples for
  occluded, clear, and out-of-map cases. The roadmap's visual bar is now
  measured rather than inspected: a `shadow_quality.h` metric API and an
  acceptance suite over a rendered four-cascade scene report zero false-shadowed
  probes of 1,520, an unbroken contact shadow, a continuous umbra transect
  across all three splits, a worst-case cascade step of 0.031 at the split
  against 1.000 unblended, and a 24-frame camera translation whose shadow-edge
  visibility moves by at most 0.0045. Each threshold is paired with a control
  configuration that fails it. The full Debug suite passes 131/131 in 169.89 s
  and the full Release suite 131/131 in 62.25 s on this PC. GPU production-scene
  PBR rendering is outside this phase's scope and remains unbuilt, the
  measurements cover the software sampler only, and no Unreal parity is
  claimed. See the
  [Phase 28 design](design/phase-28-shadows.md) and
  [implementation report](reports/phase-28.md).
- **Phase 29:** complete for its software-reference scope. It includes:
  RGBE HDR loading, an owned environment-map build path, diffuse/specular
  precomputation, PBR integration, local probe blending/parallax, hierarchical
  SSR fallback, procedural sky, specular occlusion, scene-driven six-face
  software probe capture, and measured sphere/probe/SSR acceptance. Its
  acceptance contract is the software reference renderer, not GPU scene
  rendering or cooked-cache ownership. See the
  [Phase 29 design](design/phase-29-ibl-reflections.md) and
  [implementation report](reports/phase-29.md).
- **Phase 30:** complete for its software-reference scope. The HDR post chain
  runs ambient occlusion, depth of field, motion blur, bloom, exposure, tone
  mapping, and colour grading in a fixed physical order; every pass toggles at
  runtime, is skipped rather than failed when its input is missing, and reports
  its own CPU time. Exposure meters a percentile window of a log2-luminance
  histogram and adapts at separate rates up and down. Anti-aliasing adds FXAA
  and a temporal resolver with Halton jitter, velocity reprojection, YCoCg
  neighbourhood rectification, and flicker-relaxed box widening, which is what
  lets a half-pixel feature survive rectification while a hard cut still does
  not. MSAA is not implemented and is not planned: a software rasterizer has no
  fixed-function coverage mask, and the renderer-wide supersampled path stands
  where MSAA would. `post_process_quality.h` and a rendered acceptance suite
  measure the roadmap's ghosting bar instead of inspecting it: 1.9% of the
  previous frame survives into the resolve against 48.4% for the control with
  reprojection and rectification switched off, the converged error against an
  8x8 integrated reference is 0.000943 against 0.026216 for a single sample per
  pixel, and the resolve's edge count lands on the reference's own rather than
  below it. Each threshold is paired with a configuration that fails it. GPU
  implementations of these passes are outside this phase's contract and remain
  unbuilt. The full Debug suite passes 138/138 in 192.90 s and the full
  Release suite 138/138 in 68.95 s on this PC. See the
  [Phase 30 design](design/phase-30-post-processing.md) and
  [implementation report](reports/phase-30.md).
- **Phase 31:** complete for its software-reference scope.
  [ADR 0028](adr/0028-gi-approach.md) chooses baked irradiance volumes over
  lightmaps, screen-space GI, and voxel GI. A CPU ray tracer in DXR's shape —
  bottom- and top-level acceleration structures, closest-hit and any-hit
  queries, instance masks — drives a Lambertian path tracer, which bakes a grid
  of order-2 SH probes, each with a 16x16 octahedral map of visibility-distance
  moments, and classifies probes buried in geometry. The forward PBR path takes
  the volume as its diffuse ambient term, replacing only the environment's
  irradiance inside the image-based path so the specular energy split is
  unchanged. `gi_quality.h` and a rendered acceptance suite measure the result
  against path-traced ground truth: 5.9% mean error and a 0.988 energy ratio in
  a Cornell-style room against 54.2% error for a zero-bounce control, and 0.04%
  mean light leak through a shared wall against 21.4% for the same volume
  sampled without visibility. An SVGF-style spatio-temporal denoiser takes a
  one-sample ray-traced soft shadow from 16.0% to 2.9% relative RMSE in one
  frame and 1.7% in eight. The RHI reports `Feature::RayTracing` and
  `Feature::RayQuery` from D3D12's ray-tracing tier, and
  `SelectRayTracingPlan` selects a ray-traced technique per effect only where
  they are reported, with the rasterized path as the fallback; this PC's HD
  Graphics 530 and WARP adapters both report no support and fall back. The GPU
  ray-tracing path is not implemented — it cannot be exercised on this
  hardware — and the build guide's second-machine verification remains
  outstanding. The full Debug suite passes 144/144 in 190.82 s, Release
  144/144 in 65.72 s, and Debug-ASan 144/144 in 524.72 s on this PC. See
  the [Phase 31 design](design/phase-31-gi-raytracing.md) and
  [implementation report](reports/phase-31.md).
- **Phase 32:** complete on Direct3D 12 for everything this machine can run.
  `GpuDrivenScene` describes instances, draw groups with up to four LODs, and
  their batches once; `GpuCullingPipeline` culls them on the GPU every frame —
  frustum, Phase 22's screen-space LOD rule, and optionally a Hi-Z pyramid of
  the previous frame's depth built on the GPU — appends survivors to per-batch
  visible lists, compacts non-empty batches into indirect arguments in batch
  order, and draws them all with one `DrawIndexedIndirectCount`. On the HD
  Graphics 530 a million instances are culled in two dispatches and drawn with
  one call carrying 38-40 indirect draws; a point pipeline that writes one
  texel per instance shows every drawn instance is one the culler kept.
  `CullInstancesReference` runs the shader's arithmetic on the CPU and agrees
  with Phase 22's `Scene::Cull` on every instance and LOD; the GPU agrees with
  it on every instance, with and without occlusion, on the HD Graphics 530 at
  a million and on WARP at a tenth of that. Recording a frame costs the CPU
  0.5-0.8 ms from a thousand instances to a million, against a CPU-driven
  control that grows about forty-fold over a hundredfold increase. The RHI
  gained real D3D12 indirect execution through command signatures and the
  count-buffer variant behind `Feature::IndirectDrawCount`; building it found
  and fixed three backend defects — a hard-coded triangle-list topology that
  made point and line pipelines draw nothing, missing UAV barriers between
  dispatches, and a compute-list state D3D12 rejects. Instance culling and a
  GPU cluster light-list build run together on the async compute queue behind
  a fence; `SelectAsyncComputePlan` keeps that schedule only where it measures
  faster, and on this GPU it measures within noise. The mesh-shader pipeline
  is not implemented: it needs shader model 6.5 and DXC, and neither local
  adapter reports a mesh-shader tier. Meshlets, their sphere and normal-cone
  culling, `Feature::MeshShaders`, `SelectGeometryPath`, and the
  vertex-pipeline fallback are. The full Debug suite passes 146/146 in 1053.90 s — 870.6 s of it the D3D12 GPU-driven suite under GPU-based validation, with the debug layer silent — and Release 146/146 in 78.47 s on this PC. See the
  [Phase 32 design](design/phase-32-gpu-driven.md) and
  [implementation report](reports/phase-32.md).
- **Phase 33:** the BC1, BC4, BC5, and BC7 codec and CPU-side budgeted mip
  streaming path are implemented and tested. An experimental CPU virtual
  texture cache and sample are also present. The page pool is not bound to
  D3D12 reserved resources, and streaming does not use GPU sampler feedback.
  See the [Phase 33 design](design/phase-33-compression-streaming.md) and
  [implementation report](reports/phase-33.md).
- **Phases 34-44:** not started.
- **Current PC:** Windows 10 Pro 22H2, Intel Core i5-6400 (4 cores), Intel HD
  Graphics 530 (1 GiB reported), Intel driver `31.0.101.2125` dated
  2023-05-24. The marker sample now submits and presents on this adapter after
  fixing an RTV descriptor-heap mismatch; this is not a broad hardware matrix.
- **Visual progress on this PC:** the software-rendered textured cube works.
  With the new 1x default, a two-frame 1280x720 smoke run measured 35.1 FPS; a
  deterministic 640x360 capture succeeded. This does not exercise the GPU.
- **D3D12 sample:** a native probe created a device, direct queue, command
  allocator, and command list. The sample's backbuffer RTV had used the
  shader-resource heap and the wrong heap kind; this was corrected to use the
  render-target heap. The initial `DXGI_ERROR_DRIVER_INTERNAL_ERROR`
  (`0x887A0020`) no longer reproduces, and the sample submits and presents.
- **WARP investigation:** automated WARP-backed RHI tests pass. A temporary
  windowed marker-sample attempt crashed in `d3d10warp.dll` with
  `0xC0000005`; no WARP capture was produced. This is a separate historical
  limitation, not the current Intel hardware result.
- **Capture tools:** PIX 2610.28 is installed and portable RenderDoc 1.48 is
  available outside the repository. RenderDoc shows `Clear Pass` with
  `Clear backbuffer` beneath it and the blue presented frame; PIX shows the
  nested `Clear Pass` region and the blue presented frame, while the raw native
  point-marker call remains hidden as a separate visible UI event. PIX event-list
  export requires Windows Developer Mode, which remains disabled.

## Phase-by-phase state

### Part 1 — Foundation

| Phase | Status | Delivered / remaining |
|---|---|---|
| 1. Repository and build system | Complete | CMake, presets, compiler settings, scripts, tests, and CI. |
| 2. Core utilities | Complete | Types, errors/results, assertions, logging, build configuration, and test runner. |
| 3. Math library | Complete | Vector/matrix and geometry math, transforms, conventions, and tests. |
| 4. Memory and containers | Complete | Allocators, containers, handles, and sanitizer coverage. |
| 5. Platform layer and job system | Complete | Windows platform/window services, timing, input, threads, and jobs. |

### Part 2 — CPU software renderer

| Phase | Status | Delivered / remaining |
|---|---|---|
| 6. Framebuffer and present | Complete | CPU framebuffer, conversion/present path, pacing, and frame-time display. |
| 7. Triangle rasterization | Complete | Line/triangle rasterization, top-left coverage, tiled jobs, and tests. |
| 8. Depth, clipping, and perspective | Complete | Reverse-Z depth, homogeneous clipping, camera, perspective interpolation, and tests. |
| 9. Textures, CPU shaders, and test scenes | Complete | Texture sampling, CPU shader/lighting, mesh loading, five golden images, and interactive textured-cube sample. Current machine's measured 1x smoke run: 35.1 FPS at 1280x720 for two frames. |

### Part 3 — RHI and Direct3D 12

| Phase | Status | Delivered / remaining |
|---|---|---|
| 10. RHI design | Complete | RHI object model, capabilities, backend boundary, and null backend. |
| 11. Resources | Complete | Buffers, textures, views, samplers, format table, and resource states. |
| 12. Pipelines and binding model | Complete | Shader modules, binding groups, pipeline layouts/state, and cache. |
| 13. Command lists, synchronization, swapchain | Complete | Recording, render passes, barriers, queues/fences, frame ring, retirement, and swapchain contract/tests. |
| 14. Direct3D 12 backend | Complete by recorded Phase 14 evidence | D3D12 adapter/device/queues, resources, shaders/bindings, pipelines, recording/submission, swapchain, and image comparison against software output. The Phase 15 sample-path RTV heap-kind mismatch on the Intel host is fixed; hardware coverage remains limited to this adapter. |
| 15. GPU memory and debug tooling | Complete | Heap suballocation/transient reuse, validation, object naming, native markers, timestamps, leak reporting, and device-loss diagnostics. Automated test matrix passes. RenderDoc and PIX captures confirm the nested `Clear Pass` and `Clear backbuffer` labels and the blue presented frame on the Intel HD 530 after the RTV heap-kind fix. |

### Part 4 — Shaders and materials

| Phase | Status | Intended outcome |
|---|---|---|
| 16. Shader language and compiler strategy | Complete | ADR 0019, the language design, compiler, source diagnostics, tests, and generated-HLSL cube integration are complete; Phase 16 evidence is recorded in its report. |
| 17. Reflection, permutations, and hot reload | Complete | Compiler-owned reflection, bounded/stable permutations, checksummed cache, job-system compilation, reflected renderer layouts, async frame-boundary swaps, and D3D12 live reload. All five Windows presets pass 96/96; the valid edit, syntax rejection, and restore pass in 48/49/84 ms on the Intel HD 530. |
| 18. Material system | Complete | Immutable material definitions, mutable reflected instances, texture slots, explicit uploads, sorted queues, `.cybermat` serialization, and a 100-instance D3D12 sample; evidence is in the Phase 18 report. |
| 19. Constant streaming and bindless resources | Complete | Fence-safe frame streaming, dynamic offsets, descriptor allocators, capability-gated bindless/fallback paths, and a 10,000-draw Intel HD 530 benchmark meeting the 16.67 ms frame budget; Debug, Release, Debug-ASan, package/source, and repository checks pass. See the Phase 19 report. |

### Part 5 — Renderer architecture

| Phase | Status | Intended outcome |
|---|---|---|
| 20. Render graph | Complete | Inferred dependencies and barriers, pass culling, compatible transient aliasing, bounded text/DOT output, and a tested multi-pass frame; Debug, Release, Debug-ASan, package/source consumers, and repository checks pass. See the Phase 20 report. |
| 21. Render thread and frame pipelining | Complete | Bounded frame handoff, render-owner RHI access, profiler-proven overlap, and sanitizer/stress evidence. See the Phase 21 report. |
| 22. Scene, render proxies, and culling | Complete | Copied proxies, parallel culling, screen-space LOD selection, and committed Debug/Release/ASan, sanitizer, package/source, benchmark, and repository-check evidence. See the Phase 22 report. |
| 23. Geometry pipeline | Complete | Mesh processing, packed formats, LODs, and indirect batches; Debug/Release/ASan pass 106/106, the million-triangle LOD render and RHI indirect submission pass, and package/repository checks pass. See the Phase 23 report. |
| 24. Asset import and cooked render assets | Complete | In-house importers/decoders, versioned cooked formats, asset database/cooker, hot reload, and bounded parser mutation tests. See the Phase 24 report for benchmark and local validation evidence. |
| 25. Engine-facing renderer interface | Complete | Stable versioned C plugin boundary, capabilities, scene/view/light/material API, and software-renderer implementation. |

### Part 6 — Lighting and visual quality

| Phase | Status | Intended outcome |
|---|---|---|
| 26. Physically based forward renderer | Complete | Metallic-roughness BRDF, lights, HDR, tone mapping, and reference comparison. |
| 27. Clustered lighting and deferred option | In progress | Conservative CPU cluster lists and occupancy data are implemented and tested; GPU list construction, shading integration, visual debug view, deferred comparison, and end-to-end 1,000-light acceptance remain outstanding. |
| 28. Shadows | Complete for its software scope | Software PCSS, shadowed height fog, cascades, seam blending, depth-only map rendering, CPU atlas/filtering/debug tools, and D3D12 graph-scheduled shadowed receiver lighting are tested. The acne, detached-shadow, cascade-seam, and shimmer bar is measured by a metric API and a rendered four-cascade acceptance suite, each threshold paired with a failing control. GPU production-scene integration is out of scope and remains unbuilt. |
| 29. Image-based lighting and reflections | Complete for software scope | Deterministic software IBL, RGBE HDR loading, owned environment precomputation, scene-driven six-face probe capture, PBR integration, blended/parallax-corrected probes, Hi-Z SSR fallback, procedural atmospheric sky, specular occlusion, and measured acceptance are implemented. Its contract does not include GPU scene rendering. See the Phase 29 design and report. |
| 30. Post-processing and anti-aliasing | Complete for its software scope | Histogram auto exposure with adaptation, four tone-mapping operators, baked lift/gamma/gain grading, bloom, screen-space ambient occlusion, depth of field, and motion blur, each toggled at runtime and reporting its own CPU time; FXAA and temporal anti-aliasing with jitter, velocity reprojection, and flicker-relaxed rectification. The ghosting bar is measured by a metric API and a rendered acceptance suite, each threshold paired with a failing control. MSAA is out of scope for a software rasterizer; GPU implementations remain unbuilt. |
| 31. Global illumination and ray tracing | Complete for its software scope | Baked irradiance volumes (ADR 0028) over a DXR-shaped CPU ray tracer, consumed by the forward PBR path, with an SVGF-style denoiser and measured accuracy, leakage, and noise bars. The ray-tracing capability gate falls back to the rasterized path on every adapter; the GPU ray-tracing path and second-machine verification remain outstanding. See the Phase 31 design and report. |

### Part 7 — Scale, backends, and performance

| Phase | Status | Intended outcome |
|---|---|---|
| 32. GPU-driven rendering and compute | Complete on D3D12 | GPU frustum, LOD, and Hi-Z occlusion culling with ordered compaction and one indirect-count draw; a million instances culled in two dispatches and drawn with one call, matching the CPU culler on every instance, at flat CPU cost. Real D3D12 indirect execution, async-compute culling and light culling behind a measured policy, and meshlets with a vertex-pipeline fallback. The mesh-shader pipeline is not implemented. See the Phase 32 design and report. |
| 33. Texture compression and streaming | Complete for CPU codec and streaming; experimental CPU virtual texturing | BC1, BC4, BC5, and BC7 codecs; budgeted asynchronous mip residency; virtual-texture cache/reference sample. D3D12 reserved-resource binding and GPU sampler feedback remain unimplemented. See the Phase 33 design and report. |
| 34. Vulkan backend | Not started | Second explicit backend, synchronization/presentation/validation, and Linux/Android support. |
| 35. Metal and other backends | Not started | Metal for Apple platforms; evaluate WebGPU/console only with deliberate scope/access. |
| 36. Performance hardening | Not started | CPU/GPU/memory budgets, frame-time percentiles, pipeline warmup, and CI regression benchmarks. |
| 37. Specialized rendering features | Not started | Prioritized skinning, particles, decals, terrain, water, volumetrics, vegetation, 2D/UI, and debug drawing. |

### Part 8 — Tools, quality, and shipping

| Phase | Status | Intended outcome |
|---|---|---|
| 38. Debug UI and in-engine profiling | Not started | First-party overlay, timelines, console variables, and render-graph viewer. |
| 39. Frame capture and debugging tools | Not started | First-party frame recording/replay/inspection alongside PIX and RenderDoc. |
| 40. Automated testing | Not started | Golden-image/backend matrix, fuzz/stress/long-run/crash testing, NVIDIA/AMD/Intel hardware coverage. |
| 41. Editor viewport integration | Not started | Editor rendering, picking, outlines, gizmos, previews, and thumbnails. |
| 42. Documentation, samples, and public API stability | Not started | API/tutorials, ten samples, and version/deprecation policy. |
| 43. Packaging, licensing, and distribution | Not started | SDK/installer/update/distribution, notices, and legal review. |
| 44. Beta program and release | Not started | Alpha/beta support process, compatibility matrix, release gate, and launch plan. |

## Known issues and roadmap decisions to revisit

1. **Hardware coverage is still narrow.** The clear-pass sample now submits and
   presents on the local Intel HD 530, but this single machine does not
   establish a supported adapter/driver matrix or performance targets.
2. **There is no automatic renderer fallback.** The CPU renderer is a separate
   product/sample, not a selectable backend for the D3D12 marker sample. The
   stable engine-facing selection interface is planned for Phase 25. A revised
   roadmap should decide whether a software fallback or a D3D11 backend is a
   product requirement and schedule it explicitly; neither exists today as an
   automatic fallback for this sample.
3. **The windowed WARP path remains unverified.** Tests using WARP pass, but a
   temporary windowed sample attempt crashed inside `d3d10warp.dll`. Do not rely
   on WARP as an interactive fallback based on the RHI tests alone.
4. **Capture evidence is scoped.** PIX and RenderDoc show the nested `Clear
   Pass` and `Clear backbuffer` labels and the presented blue frame. The raw
   native point-marker API is validated by RHI tests but is not exposed as a
   separate visible PIX event; this does not block Phase 15 acceptance.
5. **Current software-renderer performance is modest.** The 35.1 FPS figure is
   a short two-frame sample smoke measurement, not a benchmark. Add repeatable
   performance measurements before setting low-end device targets.
6. **Phase 40 hardware matrix is still future work.** Introduce a small
   GPU/driver compatibility test and a fallback acceptance target well before
   advanced rendering work.
7. **Scope and maturity differ.** Phases 1-25 are marked complete, but this
   repository is still a renderer foundation: full PBR lighting, editor
   integration, and a shipping workflow are not implemented.

## Verified validation and worktree notes

- On the Phase 18 implementation, Debug, Release, and Debug-ASan each passed
  **98/98** tests. The Release installed-package consumer and Debug
  source-tree consumer both built and ran.
- The D3D12 material sample self-test passed on the Intel HD Graphics 530:
  100 definitions, 100 instances, 100 sorted draws, and a live parameter
  change at frame 20.
- Structure, layer, formatting, and `git diff --check` passed.
- Phase 18 code, documentation, package rules, and validation evidence are
  committed in `679a64c` (`renderer: add Phase 18 material system`).
- Phase 21 implementation and validation evidence are recorded in the
  [Phase 21 report](reports/phase-21.md). The Linux TSan preset's Clang 21
  warning-policy exception is documented there.
- Phase 25 Windows validation passed: `cmake --build build/phase25`, then
  `ctest --test-dir build/phase25 --output-on-failure` (126/126); the
  installed-package C and C++ consumer built and ran; structure and layer
  checks passed. Targeted coverage includes the C API consumer, renderer
  runtime/fallback, software renderer, and D3D12 plugin integration.
- `diagnosing-low-phase-6-sample-frame-rate.json` is unrelated and untracked;
  leave it alone.

## Immediate next step

Continue Phase 28, shadows, using the
[master graphics roadmap](graphics/CYBERGRAPHICS_ROADMAP.md). The software path
now includes PCSS and shadowed height fog, and a D3D12 test verifies map
rasterization/readback. Complete production renderer lighting/resource
scheduling and scene-level artifact and camera-motion acceptance before marking
the phase complete. Phase 27's CPU clustered-forward path and
standalone D3D12 compute proof are documented and validated. Full renderer integration,
automatic backend switching, and deferred rendering remain future work, not
completed features.
Phase 26's forward PBR implementation and independent reference evidence are
complete.
Phase 25's plugin boundary, native surface, startup selection, and
device-creation fallback are also complete; see the [Phase 25 design note](design/phase-25-engine-interface.md)
and [plugin ABI](plugin-abi.md) for that contract.
