# CyberGraphics roadmap index

This is the master checklist for the 44-phase CyberGraphics plan. The two source
documents are binding:

- [CyberGraphics Roadmap](graphics/CYBERGRAPHICS_ROADMAP.md) — the product plan,
  phase outcomes, and done-when lines.
- [CyberGraphics Build Guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md) — the
  per-phase work packages, reading list, tasks, tips, and report template.

Work phases in this order. Do not tick a phase until its done-when line is
proved with committed evidence. Phases 1-33 are marked complete; implementation
and local validation are recorded in their phase reports.

## Part 1: Foundation (phases 1-5)

- [x] **Phase 1 — Repository and build system** — [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-1-repository-and-build-system) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-1-repository-and-build-system)
- [x] **Phase 2 — Core utilities** — [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-2-core-utilities) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-2-core-utilities)
- [x] **Phase 3 — Math library** — complete; report and CI evidence recorded · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-3-math-library) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-3-math-library)
- [x] **Phase 4 — Memory and containers** — complete; report and CI evidence recorded · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-4-memory-and-containers) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-4-memory-and-containers)
- [x] **Phase 5 — Platform layer and job system** — complete; report and CI evidence recorded — [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-5-platform-layer-and-job-system) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-5-platform-layer-and-job-system)

## Part 2: CPU software renderer (phases 6-9)

- [x] **Phase 6 — Framebuffer and present** — complete; report and CI evidence recorded · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-6-framebuffer-and-present) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-6-framebuffer-and-present)
- [x] **Phase 7 — Triangle rasterization** — complete; report and CI evidence recorded · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-7-triangle-rasterization) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-7-triangle-rasterization)
- [x] **Phase 8 — Depth, clipping, and perspective** — complete; report and CI evidence recorded · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-8-depth-clipping-and-perspective) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-8-depth-clipping-and-perspective)
- [x] **Phase 9 — Textures, CPU shaders, and test scenes** — complete; [report](reports/phase-09.md), 51/51 tests in four presets, and green CI on run 40198572801 · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-9-textures-cpu-shaders-and-test-scenes) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-9-textures-cpu-shaders-and-test-scenes)

## Part 3: RHI and Direct3D 12 (phases 10-15)

- [x] **Phase 10 — RHI design** — complete; [report](reports/phase-10.md), seven test binaries against the null backend, and the interface compiles and links · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-10-rhi-design) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-10-rhi-design)
- [x] **Phase 11 — Resources** — complete; [report](reports/phase-11.md), twelve RHI test binaries, green CI on run 40285338746, and the null backend passes creation, lifetime, and invalid-use tests · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-11-resources) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-11-resources)
- [x] **Phase 12 — Pipelines and binding model** — complete; [report](reports/phase-12.md), sixteen RHI test binaries, green CI on run 40289190526, and every pipeline described by plain structs with invalid combinations rejected · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-12-pipelines-and-binding-model) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-12-pipelines-and-binding-model)
- [x] **Phase 13 — Command lists, synchronization, swapchain** — complete; [report](reports/phase-13.md), twenty-three RHI test binaries, green CI on run 40306013967, and the missing-barrier and frame-ring tests that the done-when line names · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-13-command-lists-synchronization-swapchain) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-13-command-lists-synchronization-and-swapchain)
- [x] **Phase 14 — Direct3D 12 backend** — complete; [report](reports/phase-14.md), [design note](design/phase-14-d3d12.md), [ADR 0017](adr/0017-d3d12-backend.md), six D3D12 test binaries plus the cross-module `backend_compare`, 81/81 tests in debug, release, shipping and ASan,and the done-when line's two images matching `cyber_software_renderer`, both 4x
  supersampled and box-filtered to 512x512 (triangle exact, cube 1089 of 262144
  pixels over tolerance on silhouette edges) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-14-direct3d-12-backend) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-14-direct3d-12-backend)
- [x] **Phase 15 — GPU memory and debug tooling** — increments 1–7 and the automated build/test checks are complete. The Intel HD Graphics 530 sample-path RTV heap-kind bug is fixed; the marker sample submits and presents on the adapter. PIX and RenderDoc captures both show the nested `Clear Pass` and `Clear backbuffer` regions and the blue presented frame. The repository keeps the native D3D12 marker API in the RHI tests, but capture-visible labels come from nested debug regions because Microsoft documents the raw `SetMarker` support method as internal and recommends WinPixEventRuntime. · [implementation summary](graphics/PHASE_15.md), [report](reports/phase-15.md), [design note](design/phase-15-gpu-memory-debug.md), [ADR 0018](adr/0018-gpu-heap-suballocation-and-rhi-validation.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-15-gpu-memory-and-debug-tooling) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-15-gpu-memory-and-debug-tooling)

## Part 4: Shaders and materials (phases 16-19)

- [x] **Phase 16 — Shader language and compiler strategy** — compiler emits HLSL used by the D3D12 cube comparison; Debug, Release, and Debug-ASan pass 88/88, package/structure/layer checks pass, and the repository-wide formatter passes. · [implementation summary](graphics/PHASE_16.md), [report](reports/phase-16.md), [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-16-shader-language-and-compiler-strategy) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-16-shader-language-and-compiler-strategy)
- [x] **Phase 17 — Reflection, permutations, and hot reload** — complete; [implementation summary](graphics/PHASE_17.md), [report](reports/phase-17.md), [design note](design/phase-17-shader-permutations.md), [ADR 0020](adr/0020-shader-reflection-permutations-and-reload.md); 96/96 tests pass in Debug, Profile, Release, Shipping, and Debug-ASan, with package, structure, layer, formatting, and D3D12 live-reload checks passing · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-17-reflection-permutations-and-hot-reload) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-17-reflection-permutations-and-hot-reload)
- [x] **Phase 18 — Material system** — complete; Debug, Release, and Debug-ASan pass 98/98; package/source consumers, sample, and repository checks pass · [summary](graphics/PHASE_18.md) · [report](reports/phase-18.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-18-material-system) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-18-material-system)
- [x] **Phase 19 — Constant streaming and bindless resources** — the Intel HD Graphics 530 Release bindless path submits 10,000 unique-transform draws within the 16.67 ms budget; Debug, Release, and Debug-ASan pass 99/99 · [summary](graphics/PHASE_19.md) · [report](reports/phase-19.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-19-constant-streaming-and-bindless-resources) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-19-constant-streaming-and-bindless-resources)

## Part 5: Renderer architecture (phases 20-25)

- [x] **Phase 20 — Render graph** — Debug, Release, and Debug-ASan pass 100/100; package/source consumers and repository checks pass; the null-backend multi-pass test verifies automatic barriers, culls an unused pass, and writes a DOT dump · [summary](graphics/PHASE_20.md) · [report](reports/phase-20.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-20-render-graph) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-20-render-graph)
- [x] **Phase 21 — Render thread and frame pipelining** — profiler proves producer/render overlap; Linux ThreadSanitizer passes the null-backend render-thread and hot-pipeline targets; resize/shutdown stress passes; Debug, Release, Debug-ASan, package/source consumers, and repository checks pass · [summary](graphics/PHASE_21.md) · [report](reports/phase-21.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-21-render-thread-and-frame-pipelining) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-21-render-thread-and-frame-pipelining)
- [x] **Phase 22 — Scene, render proxies, and culling** — Debug, Release, and Debug-ASan pass 104/104; scene ThreadSanitizer passes 2/2; package/source consumers and repository checks pass; 100,000-proxy benchmark matches all 1,681 visible handles at 2.881 ms p95 · [summary](graphics/PHASE_22.md) · [report](reports/phase-22.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-22-scene-render-proxies-and-culling) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-22-scene-render-proxies-and-culling)
- [x] **Phase 23 — Geometry pipeline** — Debug, Release, and Debug-ASan pass 106/106; the 10,000-instance scene renders over one million triangles with visible LOD regions, and the RHI indirect batches are submitted and fence-verified; package consumer and repository checks pass · [summary](graphics/PHASE_23.md) · [report](reports/phase-23.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-23-geometry-pipeline) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-23-geometry-pipeline)
- [x] **Phase 24 — Asset import and cooked render assets** — Debug passes 122/122, including cooker integration and bounded fuzzing; Release, Debug-ASan, package, repository, and benchmark evidence is recorded in the report · [summary](graphics/PHASE_24.md) · [report](reports/phase-24.md) · [design](design/phase-24-assets.md) · [format](formats-cooked.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-24-asset-import-and-cooked-render-assets) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-24-asset-import-and-cooked-render-assets)
- [x] **Phase 25 — Engine-facing renderer interface** — complete; plugin runtime, software/D3D12 implementations, host fallback, and runtime compatibility validation are committed · [design](design/phase-25-engine-interface.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-25-engine-facing-renderer-interface) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-25-engine-facing-renderer-interface)

## Part 6: Lighting and visual quality (phases 26-31)

- [x] **Phase 26 — Physically based forward renderer** — complete; normal-mapped PBR sphere grid validated against an independent CPU path tracer with 8x supersampling (mean sRGB error 0.00879, coverage mismatch 0.030%); enforced limits and renderer tests pass · [summary](graphics/PHASE_26.md) · [report](reports/phase-26.md) · [design](design/phase-26-pbr.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-26-physically-based-forward-renderer) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-26-physically-based-forward-renderer)
- [x] **Phase 27 — Clustered lighting** — CPU clustered-forward PBR, 1,000-light rendered differential coverage, heat-map occupancy data, and reproducible CPU benchmark; standalone D3D12 compute proof validates GPU cluster lists and synthetic-surface PBR output against CPU references. Full rasterized GPU integration, automatic backend switching, visual overlay, and deferred comparison remain future work · [design](design/phase-27-lighting.md) · [report](reports/phase-27.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-27-clustered-lighting-and-deferred-option) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-27-clustered-lighting-and-deferred-option)
- [x] **Phase 28 — Shadows** — reverse-Z software PBR sampling with PCF, contact-hardening PCSS, point-face seam blending, practical cascades with texel snapping and overlap blending, a CPU shadow atlas, shadowed height fog, and a depth-only shadow-map raster path with no color target. The roadmap's bar is measured, not inspected: `shadow_quality.h` plus a rendered four-cascade acceptance suite report no acne, no detached shadow, continuous cascade transects, and stable shadows under camera motion, each threshold paired with a control configuration that fails it. RHI depth-only pipelines, shadow-resource lifecycle, graph scheduling, and a D3D12 comparison-sampling test are in place; production GPU shadowed scene rendering is out of scope and remains future work · [design](design/phase-28-shadows.md) · [report](reports/phase-28.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-28-shadows) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-28-shadows)
- [x] **Phase 29 — Image-based lighting and reflections** — complete for the software-reference scope: owned IBL, scene-driven six-face probe capture, blended/parallax-corrected probes, SSR fallback, procedural sky, and measured acceptance. The phase contract is software rendering and does not claim GPU-rendered Unreal Engine parity · [design](design/phase-29-ibl-reflections.md) · [report](reports/phase-29.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-29-image-based-lighting-and-reflections) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-29-image-based-lighting-and-reflections)
- [x] **Phase 30 — Post-processing and anti-aliasing** — complete for the software-reference scope: histogram auto exposure with eye adaptation, four tone-mapping operators, lift/gamma/gain grading baked to a 3D LUT, bloom, screen-space ambient occlusion, depth of field, motion blur, and FXAA and temporal anti-aliasing. Every pass toggles at runtime and reports its own CPU time, and the roadmap's ghosting bar is measured rather than inspected: `post_process_quality.h` plus a rendered acceptance suite fit 1.9% of the previous frame surviving into the resolve against 48.4% for the control that has reprojection and rectification switched off, with every threshold paired with a configuration that fails it. MSAA is not implemented: a software rasterizer has no fixed-function coverage mask, and the renderer's supersampled path stands in its place. GPU implementations of these passes are out of scope · [design](design/phase-30-post-processing.md) · [report](reports/phase-30.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-30-post-processing-and-anti-aliasing) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-30-post-processing-and-anti-aliasing)
- [x] **Phase 31 — Global illumination and ray tracing** — complete for the software-reference scope: [ADR 0028](adr/0028-gi-approach.md) ships baked irradiance volumes first; a DXR-shaped CPU ray tracer (BLAS/TLAS, closest- and any-hit queries) bakes order-2 SH probes with 16x16 octahedral visibility maps, and the forward PBR path consumes them as its diffuse ambient term. The method runs in the test scene and is measured rather than inspected: 5.9% mean error against a path-traced ground truth (energy ratio 0.988) against 54.2% for the zero-bounce control, and 0.04% mean light leak through a shared wall against 21.4% with visibility weighting off. An SVGF-style denoiser takes one-sample ray-traced shadows to 2.9% RMSE in one frame. The RHI reports `RayTracing`/`RayQuery` from D3D12 tier queries and `SelectRayTracingPlan` switches each effect to a ray-traced technique only where they are reported, falling back to the rasterized path otherwise; both local adapters report no support and fall back. The GPU ray-tracing path itself (acceleration-structure builds, dispatch, inline queries) is not implemented, and the second-machine verification on capable hardware remains outstanding · [design](design/phase-31-gi-raytracing.md) · [report](reports/phase-31.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-31-global-illumination-and-ray-tracing) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-31-global-illumination-and-ray-tracing)

## Part 7: Scale, backends, and performance (phases 32-37)

- [x] **Phase 32 — GPU-driven rendering and compute** — complete on Direct3D 12 for everything this machine can run: a compute pass frustum-culls, LOD-selects, and Hi-Z occlusion-culls every instance, an ordered compaction pass writes the indirect arguments and the draw count, and one `DrawIndexedIndirectCount` draws the result. On the HD Graphics 530 a million instances are culled in two dispatches and drawn with one call, every drawn pixel belonging to an instance the culler kept, and the GPU agrees with the CPU reference on every instance, with and without occlusion, while the reference agrees with Phase 22's `Scene::Cull`. Recording a frame costs the CPU 0.5-0.8 ms from a thousand instances to a million, against a CPU-driven control that grows about forty-fold over a hundredfold increase. The RHI gained real D3D12 indirect execution and a count-buffer variant; culling and GPU light culling run together on the async compute queue behind a fence, kept only where measured faster, which on this GPU it is not. The mesh-shader pipeline is not implemented — it needs shader model 6.5 and neither local adapter reports a mesh-shader tier — though meshlets, their culling, the capability gate, and the vertex-pipeline fallback are · [design](design/phase-32-gpu-driven.md) · [report](reports/phase-32.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-32-gpu-driven-rendering-and-compute) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-32-gpu-driven-rendering-and-compute)
- [x] **Phase 33 — Texture compression and streaming** — complete for the codec and the CPU side of streaming: BC1, BC4, BC5 and BC7 decoders define the reference output, and the encoders for the same formats are held to PSNR floors in CI (BC7 43.5 dB, BC4/BC5 51.7 dB, BC1 37.4 dB on the test card). A residency manager streams mips into a 64 KiB page pool under a fixed budget, coarsest first above a packed mip tail that is never evicted and serves as the placeholder, on below-normal-priority IO threads, evicting surplus and less important levels by a plan that never thrashes. A corridor of 400 textures, 1.07 GiB of BC1 against a 64 MiB budget, streams for 1,800 frames with a worst streaming update of 1.2 ms, the budget full and never exceeded, and every texture drawable on every frame; the residency view shows what is loaded. The software renderer also got a profiling pass: per-draw page-fault churn, CRT calls in hot loops, a per-frame sRGB table rebuild, and a job-system use-after-return were fixed, roughly halving frame time at low resolution. Not done: binding the pool to D3D12 reserved resources, and virtual texturing · [design](design/phase-33-compression-streaming.md) · [report](reports/phase-33.md) · [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-33-texture-compression-and-streaming) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-33-texture-compression-and-streaming)
- [ ] **Phase 34 — Vulkan backend** — [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-34-vulkan-backend) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-34-vulkan-backend)
- [ ] **Phase 35 — Metal and other backends** — [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-35-metal-and-other-backends) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-35-metal-and-other-backends)
- [ ] **Phase 36 — Performance hardening** — [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-36-performance-hardening) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-36-performance-hardening)
- [ ] **Phase 37 — Specialized rendering features** — [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-37-specialized-rendering-features) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-37-specialized-rendering-features)

## Part 8: Tools, quality, and shipping (phases 38-44)

- [ ] **Phase 38 — Debug UI and in-engine profiling** — [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-38-debug-ui-and-in-engine-profiling) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-38-debug-ui-and-in-engine-profiling)
- [ ] **Phase 39 — Frame capture and debugging tools** — [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-39-frame-capture-and-debugging-tools) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-39-frame-capture-and-debugging-tools)
- [ ] **Phase 40 — Automated testing** — [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-40-automated-testing) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-40-automated-testing)
- [ ] **Phase 41 — Editor viewport integration** — [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-41-editor-viewport-integration) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-41-editor-viewport-integration)
- [ ] **Phase 42 — Documentation, samples, and public API stability** — [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-42-documentation-samples-and-public-api-stability) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-42-documentation-samples-and-public-api-stability)
- [ ] **Phase 43 — Packaging, licensing, and distribution** — [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-43-packaging-licensing-and-distribution) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-43-packaging-licensing-and-distribution)
- [ ] **Phase 44 — Beta program and release** — [roadmap](graphics/CYBERGRAPHICS_ROADMAP.md#phase-44-beta-program-and-release) · [build guide](graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-44-beta-program-and-release)

## Current position

- Phase 1: complete.
- Phase 2: complete.
- Phase 3: complete. Report and CI evidence recorded.
- Phase 4: complete. Report and CI evidence recorded.
- Phase 5: complete. Report and CI evidence recorded.
- Phase 6: complete. Report and CI evidence recorded.
- Phase 7: complete. Report and CI evidence recorded.
- Phase 8: complete. Report and CI evidence recorded.
- Phase 9: complete. 51/51 tests in debug, release, shipping, and ASan, five
golden scenes, a textured-lit sample captured with 4x supersampling, and all
five CI jobs green.
- Phase 10: complete. The RHI object model, capabilities, conventions, backend
boundary, and null backend; 58/58 tests in debug.
- Phase 11: complete. Resources, views, samplers, the format table, and explicit
  state transitions, validated by the null backend.
- Phase 12: complete. Shader modules, the four-slot binding model, pipeline
  layouts with push constants, graphics and compute pipelines, and a hash-keyed
  pipeline cache.
- Phase 13: complete. Command lists, render passes, submission with timeline
  fences, the frame ring, deferred destruction, and the swapchain. A missing
  barrier is caught at record time with the resource named, and the frame ring
  proves a slot is not reused before its fence.
- Phase 14: complete. Adapter discovery, the device, queues and timeline fences,
  resources, HLSL shaders and the binding model, pipeline state with pipeline
  libraries, command recording and submission, a flip-model swapchain, and the
  comparison that renders a colored triangle and a textured cube through the RHI
  and matches the software renderer's images within a recorded tolerance. The
  debug layer is enabled with GPU-based validation and its messages fail a test.
- Current: Phase 22, scene, render proxies, and culling. Its implementation and
  validation evidence are committed in the report. Phase 21's render-thread pipeline and validation
  evidence are recorded in its report. Phase 20's render graph,
  multi-pass tests, and validation evidence are recorded in its report. Phase
  19's constant streaming, descriptor management, capability-gated bindless
  path, and measured 10,000-draw
  acceptance result are recorded in its report. Phase 18's material system
  implementation and validation evidence are committed, and the master
  roadmap marks it complete. Phase 15's automated checks and capture evidence
  are complete; the Intel HD
  Graphics 530 marker sample submits and presents, and both PIX and RenderDoc
  show the nested pass labels and blue frame. The raw native point-marker call
  is covered by RHI tests; it is not presented as a separate capture UI event.
