# Changelog

All notable changes to CyberGraphics are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
semantic versioning once a public API exists.

## [Unreleased]

### Added

- Phase 32 GPU-driven rendering in `cyber_renderer`:
  `cyber/renderer/gpu_driven.h` describes a scene once — instances with world
  bounds, draw groups with up to four LODs, each LOD one indexed draw — and
  `cyber/renderer/gpu_culling_pipeline.h` culls it on the GPU every frame: a
  compute dispatch frustum-culls every instance, selects its LOD with Phase 22's
  screen-space rule, optionally tests it against a Hi-Z pyramid of the previous
  frame's depth, and appends it to its batch's slice of a visible list; a
  single-group dispatch compacts non-empty batches into indirect arguments in
  batch order; one `DrawIndexedIndirectCount` draws them. A million instances
  are culled in two dispatches and drawn with one call on the HD Graphics 530,
  and the recording cost stays at 0.5-0.8 ms from a thousand instances to a
  million.
- `CullInstancesReference` and `DecideGpuInstance`: the culling shader's
  decisions on the CPU, operation for operation, with how close each came to
  the other answer. The reference agrees with Phase 22's `Scene::Cull` on every
  instance and LOD, and the GPU agrees with the reference on every instance on
  both local adapters, with and without occlusion.
- `HiZPyramid`, `ComputeHiZLayout`, and `TestHiZOcclusion`: a conservative
  reverse-Z depth pyramid with ceiling-sized levels, built on the CPU or on the
  GPU (`GpuCullingPipeline::RecordHiZBuild`, bitwise equal to the CPU build),
  and a four-read occlusion test whose verdicts are checked against level-0
  depth under every pixel a box covers.
- `cyber/renderer/gpu_light_culling.h`: `GpuLightCullingPass` builds Phase 27's
  cluster light lists in a compute shader as a conservative superset of
  `BuildClusterLightLists`, on either queue.
- `SelectAsyncComputePlan`: culling and light culling move to the compute queue
  only when the device has one and, under `Auto`, a measurement of both
  schedules shows at least a 5% gain. On the HD Graphics 530 the measured
  difference is within noise and `Auto` keeps the graphics queue.
- `cyber/renderer/meshlet.h`: `BuildMeshlets` (64 vertices, 124 triangles),
  sphere and normal-cone culling checked by brute force against every
  triangle, and `ExpandMeshletsToIndices`, the vertex-pipeline fallback.
  `SelectGeometryPath` resolves a mesh-shader preference against the reported
  capability; no mesh-shader pipeline is compiled in, because the RHI's HLSL
  compiler tops out at shader model 5.1, so every adapter currently resolves to
  the vertex pipeline with a reason.
- RHI: `CommandList::DrawIndirectCount` and `DrawIndexedIndirectCount`, which
  read the draw count from a GPU buffer and clamp it to a CPU-side maximum,
  behind the new `Feature::IndirectDrawCount`, on the null, validating, and
  D3D12 backends.
- RHI: the D3D12 backend records `DrawIndirect`, `DrawIndexedIndirect`, and
  `DispatchIndirect` as `ExecuteIndirect` through per-device command
  signatures, instead of returning `Unsupported`, and reports
  `Feature::IndirectDraw`, `MultiDrawIndirect`, and `IndirectDrawCount`.
- RHI: `Feature::MeshShaders`, reported by the D3D12 backend from
  `D3D12_FEATURE_D3D12_OPTIONS7`.
- `samples/17_gpu_driven`: a 1000 x 1000 voxel landscape — a million
  instances — culled and LOD-selected on the GPU and drawn with one
  indirect-count draw call.

- Phase 31 ray tracing in `cyber_software_renderer`:
  `cyber/software/raytrace.h` adds a CPU ray tracer in DirectX Raytracing's
  shape. `BottomLevelAccelerationStructure` builds a binned
  surface-area-heuristic BVH over object-space triangles;
  `TopLevelAccelerationStructure` places them in the world with invertible
  transforms, identifiers, and eight-bit masks. Queries take a `RayDesc` and
  are closest-hit (`Intersect`, `TraceRay`) or any-hit (`IntersectAny`,
  `TraceVisibility`). Builds are deterministic, keep degenerate triangles so
  primitive indices never shift, and bound the tree depth so every query runs
  on a fixed stack. Normals are carried through the inverse transpose, so a
  non-uniform scale is handled correctly.
- Phase 31 baked global illumination in `cyber_software_renderer`:
  `cyber/software/global_illumination.h` adds a Lambertian path tracer with
  next-event estimation and Russian roulette (`TracePathRadiance`,
  `EstimateIrradiance`), and `IrradianceVolume`, a grid of order-2
  spherical-harmonic probes each carrying a 16x16 octahedral map of
  visibility-distance moments. `BakeIrradianceVolume` classifies probes buried
  in geometry and is transactional; `SampleIrradianceVolume` weights the eight
  surrounding probes by position, a smooth back-face term, and Chebyshev
  visibility, which is what stops light leaking through walls. The bake is
  byte-identical with and without a job system. ADR 0028 records why baked
  volumes ship before lightmaps, screen-space GI, or voxel GI.
- `PbrLightingData::irradianceVolume`: the forward PBR shader takes a baked
  volume as its diffuse ambient term. Inside the image-based path it replaces
  only the environment's irradiance, through the new
  `IblSurface::irradianceOverride`, so probes, screen-space reflections, and
  the global environment keep the specular term and its energy split.
- `cyber/software/denoise.h`: `SpatioTemporalDenoiser`, an SVGF-style filter
  for one-sample ray-traced signals — depth- and normal-tested temporal
  reprojection with luminance moments, a spatial variance estimate for short
  history, and four passes of edge-aware à-trous filtering.
- `cyber/software/gi_quality.h`: `MeasureIrradianceError`,
  `MeasureLightLeak`, `MeasureNoise`, and `MeasureLaplacianEnergy`, and an
  acceptance suite that pairs each threshold with a control configuration that
  fails it. The bake lands within 5.9% mean error of path-traced ground truth
  (54.2% for a zero-bounce control), the visibility test holds light leakage
  through a wall to 0.04% (21.4% with it off), and the denoiser takes a
  one-sample ray-traced shadow from 16.0% to 2.9% relative RMSE in one frame.
- `rhi::Feature::RayTracing` and `rhi::Feature::RayQuery`, reported by the
  D3D12 backend from `D3D12_FEATURE_D3D12_OPTIONS5` at tiers 1.0 and 1.1. They
  describe the adapter; this RHI has no acceleration-structure or ray-dispatch
  objects yet, and the null backend reports neither.
- `cyber/renderer/raytracing_path.h`: `SelectRayTracingPlan` resolves a
  ray-tracing policy against reported capabilities per effect — shadows,
  reflections, global illumination, ambient occlusion — selecting a hardware
  technique only where the adapter reports support and the build contains the
  GPU path, and the rasterized technique otherwise, with a reason for every
  fallback. No GPU ray-tracing path is implemented in this release, so every
  adapter currently resolves to the rasterized path.
- `samples/14_global_illumination`: a Cornell-style room baked into an
  irradiance volume and rendered with and without it, with ray-traced lamp
  shadows; the denoiser on one-sample soft shadows against a 256-sample
  reference; and the ray-tracing policy's answer for every local D3D12
  adapter.
- `samples/13_showcase`: every renderer feature so far in one frame -- PBR
  direct lighting, PCSS soft shadows, a reflection probe captured in the room
  and prefiltered for image-based lighting, a metal roughness sweep reading
  that probe, mipped textures at a grazing angle, 4x4 supersampled
  anti-aliasing, and the full post chain. It writes the frame twice, with the
  chain and with tone mapping alone, and prices each post pass.
- Phase 30 HDR post-processing in `cyber_software_renderer`:
  `cyber/software/post_process.h` adds `PostProcessStack`, which owns the
  scratch images for one resolution and runs ambient occlusion, depth of
  field, motion blur, bloom, exposure, tone mapping, and colour grading in a
  fixed physical order. Exposure meters a 64-bin log2-luminance histogram
  between two percentiles and adapts at separate rates up and down; tone
  mapping offers clamp, Reinhard, ACES, and Hable; grading bakes lift, gamma,
  gain, contrast, saturation, and white balance into a 3D lookup table.
  Every pass toggles at runtime, a disabled pass or one whose input is missing
  is skipped rather than failed, and each reports its own CPU time so the
  chain can be priced stage by stage. Both paths write identical bytes with
  and without a job system.
- Phase 30 anti-aliasing in `cyber_software_renderer`:
  `cyber/software/temporal_aa.h` adds Halton camera jitter, projection
  jittering, `TemporalAaResolver` for velocity-reprojected accumulation with
  YCoCg neighbourhood rectification and disocclusion rejection, and
  `FxaaResolve` as the cheap alternative and the no-history fallback. MSAA is
  not implemented: a software rasterizer has no fixed-function coverage mask,
  and `AntiAliasMethod::Supersample` selects the renderer-wide supersampled
  path that stands in its place.
- Flicker-relaxed history rectification. A neighbourhood clamp cannot preserve
  a feature narrower than a pixel: half the frames the jittered sample misses
  it, the whole neighbourhood agrees nothing is there, and the correct
  converged coverage is clamped away. The resolver now tracks a running mean
  of the signed difference between the input and its reprojected history and a
  running mean of that difference's magnitude, and widens the clip box by the
  part of the disagreement that alternates rather than points one way. A cut
  still loses its history, because the record is read before the current
  frame's difference is folded into it. On the acceptance scene this cut the
  converged error against the integrated reference from 0.015704 to 0.000943
  and brought the resolve's edge statistics onto the reference's own.
- Depth-based disocclusion rejection in the temporal resolve.
  `TemporalAaResolver::Resolve` takes a `TemporalAaInputs` with an optional
  depth buffer; the resolver keeps the previous frame's linearized view
  distance and drops history wherever the reprojected distance disagrees with
  the surface actually rendered. A velocity buffer cannot see a disocclusion:
  a pixel of floor an occluder was covering carries the floor's velocity,
  reprojects onto the occluder, and reuses its colour, which reads as the
  occluder trailing along the motion. `samples/12_post_process` showed that
  trail; comparing depth removes it, rejecting 2.5% of pixels on the sample's
  occluder edges. `TemporalAaStats::depthRejectedPixels` reports the count.
- Phase 30 image measurements: `cyber/software/post_process_quality.h` adds
  `MeasureImageDifference`, `MeasureGhosting`, `MeasureTemporalStability`, and
  `MeasureEdgeAliasing`, so the roadmap's "no visible ghosting" bar is a number
  with a threshold and a failing control rather than an inspection.
  `GhostingMetrics::weightedContamination` fits one scalar over the whole image
  — the least-squares fraction of the previous frame that survives into the
  resolve — which weights each pixel by how far it actually moved, unlike the
  per-pixel ratio that an unweighted mean lets near-static pixels dominate.
- `samples/12_post_process`: a scene with a crease for occlusion, posts past
  the focal plane, a screen of half-pixel slats, and a highlight above the tone
  mapper's shoulder. It prints the per-pass CPU time, what each toggle is
  responsible for, and the edge statistics of the unfiltered, FXAA, and
  temporal resolves of the same moving scene.
- Radiance RGBE HDR image loading in `cyber_software_renderer`: both flat and
  scanline-RLE data decode into linear RGBA32F without clipping HDR radiance.
  Decoded frames can be projected directly to cubemaps, so HDR environments
  load without an intermediate 8-bit texture.
- Phase 29 software IBL and reflections: deterministic cubemaps, solid-angle
  diffuse irradiance, GGX specular prefiltering, a split-sum BRDF LUT,
  order-2 spherical harmonics, probe blending and box parallax correction,
  Hi-Z-accelerated screen-space reflections with probe fallback, a
  Rayleigh/Mie procedural sky, and specular occlusion. `EnvironmentMap`
  assembles owned lighting resources from an image, file, or cubemap.
  `CaptureReflectionProbe` renders six caller-provided software scene views
  into a transactional HDR cubemap. The rendered sphere, white-furnace,
  probe-path, SSR, and scene-capture tests are in place. The phase's published
  contract is the software-reference renderer.
- Renderer-wide anti-aliasing in `cyber_software_renderer`:
  `cyber/software/antialias.h` adds `AntiAliasTarget`, an N x N supersampled
  color and depth pair, and `ResolveSupersampled`, a separable, job-parallel,
  banded resolve with box, tent, Mitchell-Netravali, and Blackman-Harris
  reconstruction filters. Filtering runs in linear light with normalized
  weights and edge clamping, so a constant image resolves to that exact
  constant and no border darkening appears. The renderer-wide default is
  `AntiAliasQuality::High` (4x4) with Mitchell. Resolved output is
  byte-identical with and without a job system and for any band size.
  Geometry edges, shadow edges, specular highlights, and texture minification
  are all covered, because every sample runs the full pixel stage.
- `samples/10_antialiasing`: a PBR scene of thin rods, a long railing, a metal
  torus, a roughness sphere row, and a receding tiled floor under cascaded
  shadows, captured at every quality preset and filter. It reports a hard-step
  fraction and the RMS sRGB error against the 8x8 capture: 12.00 at one sample
  per pixel, 3.61 at 2x2, and 1.58 at the 4x4 default.
- `samples/03_depth` and `samples/04_textured` captures now resolve through the
  shared path and default to 4x4; `04_textured` gains `--filter`, and
  `04_lighting` drops its private box downsampler for the shared one.

- Phase 27 groundwork: a deterministic, allocator-backed CPU clustered-light
  list builder, per-tile occupancy data, sampled-sphere correctness tests, and
  a 1,000-moving-light CPU-stage benchmark. GPU construction and forward
  shading integration remain in progress; see the
  [Phase 27 report](docs/reports/phase-27.md).
- Phase 26 complete: Cook-Torrance metallic-roughness CPU shading, tangent-
  space normal mapping, directional/point/spot lighting, ACES tone mapping,
  white-furnace coverage, and a 7x7 sphere grid validated against an independent
  deterministic CPU path tracer. The sample defaults to 8x supersampling and
  measures 0.00879 mean sRGB error and 0.030% coverage mismatch; it enforces
  documented acceptance limits. See the [Phase 26 summary](docs/graphics/PHASE_26.md)
  and [validation report](docs/reports/phase-26.md).
- Phase 22 implements `cyber_scene`: copied mesh, light, and camera proxies
  sent through the existing render-thread packet queue; generation-checked
  handles; contiguous bounds and transforms; parallel frustum culling; and
  screen-space LOD selection. Debug, Release, and Debug-ASan pass 104/104; the
  TSan scene tests, package/source consumers, and repository checks pass. The
  100,000-proxy benchmark matches brute force at 2.881 ms p95 on a 4-core
  Intel Core i5-6400. The master-roadmap checkbox is checked; see the
  [Phase 22 summary](docs/graphics/PHASE_22.md) and
  [report](docs/reports/phase-22.md).
- Phase 23 geometry pipeline: validated allocator-backed indexed meshes,
  submeshes and bounds; tangent generation; checked packed vertex/index
  conversion; deterministic edge-collapse LOD generation; and instanced
  indirect batches. The RHI integration submits batch arguments, and the
  software reference renderer produces a visible LOD transition with over one
  million triangles. Debug, Release, and Debug-ASan pass 106/106, with package
  consumer and repository checks passing. See the
  [Phase 23 summary](docs/graphics/PHASE_23.md) and
  [report](docs/reports/phase-23.md).
- Phase 21 render-thread frame pipelining: bounded SPSC command handoff,
  copied immutable packets, ordered resize and draining shutdown, profiler
  scopes, single-thread reference mode, and render-owner RHI lifetime for hot
  pipelines. Debug, Release, and Debug-ASan pass 101/101; the Linux
  ThreadSanitizer render-thread and hot-pipeline targets pass; installed and
  source consumers and structure, layer, formatting, and whitespace checks
  pass. See the Phase 21 summary and report.
- Phase 20 render graph: setup/execute pass declarations, inferred resource
  hazards and culling, graph-owned transient buffer/texture lifetimes,
  descriptor-safe alias slots, automatic RHI barriers, and bounded text/DOT
  diagnostics. Debug, Release, and Debug-ASan pass 100/100; installed-package
  and source-tree consumers, structure, layer, formatting, and whitespace
  checks pass. The null-backend multi-pass proof generates a graph DOT file.
  See the Phase 20 summary and report.
- Phase 19 RHI dynamic buffer offsets, mutable bind groups, read-only storage
  bindings and descriptor-array limits; renderer frame-slot streaming,
  descriptor allocators, and capability-gated bindless tables; and a
  10,000-draw D3D12 sample. Debug, Release, and Debug-ASan pass 99/99, with
  package/source consumers and repository checks passing. On Intel HD Graphics
  530, the Release bindless path completes the 10,000-draw frame in 11.6 ms p95,
  within the 16.67 ms budget. See the Phase 19 summary and report.
- Phase 18 material system: immutable reflected definitions, mutable
  per-instance parameter blocks and texture bindings, explicit GPU uploads,
  deterministic material draw sorting, bounded `.cybermat` serialization, and
  a 100-instance D3D12 sample with a live parameter-change self-test. Debug,
  Release, and Debug-ASan each pass 98/98; package/source consumers and
  structure, layer, formatting, and whitespace checks pass. The phase report
  and evidence are included in this commit.
- Phase 17 implementation and local validation: compiler-owned shader
  reflection, bounded keyword permutations, checksummed on-disk cache,
  job-system compilation and file watching, renderer-side reflected layouts and
  asynchronous frame-boundary pipeline swaps, and the D3D12 live-reload sample.
  All five Windows presets pass 96/96 tests; the package consumer and live
  D3D12 edit/error/restore self-test pass.
- Phase 16 complete: `cyber_shaders` compiles a bounded typed Cyber shader
  language to HLSL with source diagnostics, and the D3D12 cube comparison uses
  the generated stages. Debug, Release, and Debug-ASan pass 88/88; package,
  structure, layer, and repository-wide formatting checks pass.
- Phase 15 GPU memory and debug tooling: D3D12 heap suballocation and transient
  reuse, RHI validation, object-name read-back, first-party native command-list
  markers, fence-safe timestamps, named leak reporting, and sticky device-loss
  diagnostics. The marker sample and phase report are included. Automated
  Debug, Profile, Release, Shipping, ASan, structure, layer, formatting, and
  Profile package-consumer checks pass. PIX and RenderDoc captures show the
  nested pass labels and blue presented frame on the Intel HD Graphics 530
  after fixing the sample's RTV heap-kind mismatch. The native point-marker
  path is covered by RHI tests; PIX does not show it as a separate UI event.
- Phase 14, increment 7 (the phase's done-when line): an image comparison that
  renders the same scene through the Direct3D 12 backend and through
  `cyber_software_renderer` and compares the two images with a recorded tolerance
  — at most 5% of pixels may differ by more than 4 per channel, and the mean
  channel delta must stay under 6. It lives in
  `tests/integration/backend_compare/` because it is the one target that may link
  both renderers, and it writes its four images to
  `build/<preset>/backend_compare_images/` on every run. Both images are
  antialiased by 4x supersampling box-filtered down to 512x512, the path ADR 0012
  places at the caller until a rasterizer grows multisample coverage: the
  colored triangle matches the reference exactly and the textured cube differs
  on 1089 of 262144 pixels, all on silhouette and texel edges where the two
  rasterizers disagree about coverage. The images the test exports carry the
  engine's sRGB transfer function, because the framebuffers store linear bytes
  and the comparison measures them before any encoding is applied.
- Phase 14: every debug-layer message the backend stored reaches the `Rhi` log
  channel when the device is destroyed, at error or warning level, with the two
  messages this backend provokes by design filtered at the queue. That is what
  makes the build guide's "treat any message as a test failure" a test:
  `d3d12_commands_leave_the_debug_layer_silent` fails on a single record.
- Phase 14, increment 6: Direct3D 12 command recording, submission, and a
  flip-model swapchain. `CommandListInterface` is implemented over a real
  `ID3D12GraphicsCommandList` with every recording rule the contract states —
  one open pass, one open pipeline of the matching kind, bind groups checked
  against the layout, vertex and index buffers checked against the states they
  must be in, and an ended list as the only thing a queue accepts. `submit`
  applies GPU waits, runs the batch, and reaches its signals. A swapchain adopts
  its images into the device's texture and view tables and presents through
  `IDXGISwapChain3`.
- Attachment descriptors got the heaps the API requires: render-target and depth
  views now live in dedicated RTV/DSV heaps, because Direct3D 12 refuses them in
  a shader-visible heap. Samplers similarly take a block of their own, so a bind
  group is two descriptor ranges with a root parameter each.

- Phase 14, increment 5: Direct3D 12 graphics and compute pipeline state.
  `GraphicsPipelineDesc` becomes a `D3D12_GRAPHICS_PIPELINE_STATE_DESC` with
  every state translated — vertex layout, blend per attachment, rasterizer,
  depth and stencil, multisampling, and each attachment format — and a compute
  description becomes `D3D12_COMPUTE_PIPELINE_STATE_DESC`. Pipelines report the
  stable hash of the description they were built from.
- Direct3D 12 pipeline caches over `ID3D12PipelineLibrary`: a description that a
  previous run stored is loaded instead of compiled, seed bytes the device cannot
  use are ignored as the contract requires, and `PipelineCacheBytes` serializes
  through the library's own size query.
- `Feature::Compute` is reported by the Direct3D 12 backend, because every
  Direct3D 12 device has a compute queue and compute pipelines.

- Phase 11 `cyber_rhi` resources: `BufferUsage`/`BufferDesc`/`BufferInfo`,
  `TextureDimension`/`TextureUsage`/`TextureAspect`/`SubresourceRange`,
  `TextureDesc`/`TextureViewDesc`, and `SamplerDesc` with every filter, address,
  border, LOD, anisotropy, and comparison field a D3D12 or Vulkan sampler needs.
- A numeric format table: `FormatInfo`, `FormatInfoOf`, per-field accessors,
  sRGB/linear variant lookup, view compatibility, and multisample, render-target,
  storage, and blend support rules. `docs/formats.md` renders the table and a
  test walks every enumerator.
- A resource state vocabulary (`ResourceState`, `BufferBarrier`,
  `TextureBarrier`, `ResolveSubresourceRange`) that maps to D3D12 resource states
  and Vulkan image layouts, with read states that combine and at most one write.
- `Device` resource creation and destruction for buffers, textures, views, and
  samplers; `WriteBuffer`/`ReadBuffer` host access; `BufferStateOf` and
  `TextureStateOf`; and `Transition`, which validates a whole batch before
  applying any of it.
- Validation of every descriptor in the null backend: usage combinations,
  extents, mip chains, cubes, arrays, multisampling, limits, format capability,
  view compatibility, sampler LOD and anisotropy, and stale-handle rejection.
  Views are retired with their texture, and host-visible bytes actually
  round-trip.
- `rhi_format_table`, `rhi_buffer`, `rhi_texture`, `rhi_sampler`, and
  `rhi_resource_state` test binaries; the installed-package and source consumers
  exercise buffers, textures, views, samplers, and barriers.
- ADR 0014, the Phase 11 design note, `docs/formats.md`, and the Phase 11 report.
- Phase 12 `cyber_rhi` pipelines and binding: shader modules with stage,
  bytecode format, and entry point; four binding slots fixed by change frequency;
  group layouts with type, visibility, and array count; bind groups with
  flattened resources; pipeline layouts with group slots and up to four
  push-constant ranges; complete graphics and compute pipeline descriptions;
  and a pipeline cache object.
- Stable `HashGraphicsPipelineDesc`/`HashComputePipelineDesc` over every field
  that changes the compiled object, excluding labels.
- Validation of shader stages, layout entries and collisions, group resource
  counts and types, buffer ranges and alignment, texture and sampler usage,
  vertex bindings and attributes, rasterizer, blend, multisample, and
  depth-stencil state, attachment formats, and sample counts.
- `R32G32B32Float` joins the format table for three-component vertex positions.
- `rhi_shader`, `rhi_binding`, `rhi_pipeline`, and `rhi_pipeline_hash` test
  binaries; the package consumer creates a shader, a layout, a group, and a
  graphics pipeline through the exported target.
- ADR 0015, the Phase 12 design note, and the Phase 12 report.
- Phase 13 `cyber_rhi` command recording: `CommandList` with `Begin`/`End`/
  `Reset`, render pass begin and end, pipeline and group binding, vertex and
  index buffers, viewport and scissor, draws, indexed draws, indirect draws,
  dispatches, buffer and texture copies, color and depth clears, in-list
  barriers, push constants, and balanced debug labels.
- `RenderPassDesc` with load and store operations, reverse-Z clear values, resolve
  targets, read-only depth rules, and a pure validator both the RHI and the
  backend call.
- `SubmitDesc` with one-call submission per queue, timeline `FenceSignal` and
  GPU-side `GpuWait` entries, and validation that refuses a deadlock expressed
  as data.
- `FrameRing`, a policy object that hands out frame slots and fence values and
  waits on the slot being reused, so no resource is touched by two frames at
  once.
- `RetirementQueue`, deferred destruction keyed by fence value for every object
  kind, with collection and flush paths.
- `SwapChain` with acquire, present, resize, fifo/mailbox/immediate present modes,
  device-owned images, and stale image handles after a resize.
- Null-backend command recording, submission, and swapchain: a per-list shadow of
  resource state that validates every command in order, clear messages that name
  the missing barrier, and submission that applies waits, reaches signals, and
  commits a list's transitions when it ends.
- `rhi_render_pass`, `rhi_command_list`, `rhi_submit`, `rhi_missing_barrier`,
  `rhi_frame_ring`, `rhi_retirement`, and `rhi_swapchain` test binaries; the
  package consumer records a frame, submits it, acquires and presents an image,
  and proves a missing barrier is caught through the installed package.
- ADR 0016, the Phase 13 design note with the frame lifecycle diagram, and the
  Phase 13 report.
- Phase 14 `cyber_rhi` Direct3D 12 backend, first increment: `cyber/rhi/d3d12.h`
  with `IsD3D12CompiledIn()` and `D3D12Backend()`, DXGI adapter enumeration with
  vendor, device, and memory fields, Microsoft's WARP adapter appended last so a
  machine with no GPU is still testable, and a build option
  (`CYBER_RHI_ENABLE_D3D12`) that compiles the backend out on other platforms.
- `rhi_d3d12_adapters` test binary: adapter enumeration, the zero-capacity and
  short-span contract, WARP ordering, backend identity, and the reported error an
  incomplete backend table produces instead of a crash.
- ADR 0017 and the Phase 14 design note with the RHI-to-D3D12 mapping table and
  the phase's increment plan.
- Phase 14 `cyber_rhi` Direct3D 12 device: creation on a chosen adapter through
  `D3D12CreateDevice`, the debug layer plus GPU-based validation in debug builds,
  three command queues (direct, compute, copy), timeline fences over
  `ID3D12Fence` with queue-side signals and event-based waits that wake on
  destroy, `Capabilities` from the feature level and the API's guaranteed limits,
  and `WaitIdle` draining every queue that accepted work.
- `rhi_d3d12_device` test binary: adapter identity against enumeration, the three
  queues, the fence timeline including stale handles and slot reuse, two devices
  sharing nothing, the `Unsupported` answer an unimplemented call gives, and full
  allocator accounting on teardown.
- Phase 14 `cyber_rhi` Direct3D 12 resources: buffers placed on the heap their
  usage names (default, upload, or readback) with persistent host mapping,
  textures for every dimension the RHI describes, views that become render
  target, depth stencil, or shader resource descriptors, samplers on a
  shader-visible heap, and the barrier path with one-shot submission and
  per-subresource texture state.
- Phase 14 `cyber_rhi` Direct3D 12 shaders and binding model: HLSL source
  compiled to bytecode through the SDK's `D3DCompile` with a shader-model target
  per stage, precompiled blobs accepted verbatim, bind group layouts turned into
  descriptor tables by register space, bind groups filling one contiguous block
  of the shader-visible heap with constant buffer, structured buffer, sampled
  texture, storage texture, and sampler descriptors, and pipeline layouts
  serialized into a root signature with root constants for push constants.
- `rhi_d3d12_binding` test binary: a compiling shader, a shader that fails to
  compile, incomplete descriptions, precompiled bytecode, layouts and groups with
  a count mismatch and a stale layout, a root signature from a pipeline layout,
  and bytecode released on destroy.
- `rhi_d3d12_resources` test binary: host-visible write ranges, readback reads,
  texture and view lifetime including views retired with their texture, a format
  with no native equivalent, sampler creation, buffer and texture state across
  transitions, a mismatched and a stale barrier, and full release on teardown.
- Repository skeleton: `engines/`, `include/`, `tools/`, `samples/`, `tests/`,
  `docs/`, `cmake/`, `scripts/`, `assets/`, and `third_party/`.
- Sixteen engine modules with written contracts in `engines/MODULES.md` and
  `engines/<module>/docs/MODULE.md`.
- Stable product contract location at `include/cybergraphics/`.
- CMake 3.28+ build with Debug, Release, Shipping, ASan, and CI presets.
- Module helper (`cyber_add_module`, `cyber_add_test`) and compiler policy.
- Structure and dependency-law checks (`scripts/check_structure.py`,
  `scripts/layer_check.py`).
- Windows and Linux CI workflow.
- Architecture, maths conventions, coding standard, ADR, testing, and
  performance documentation.
- Project requirements note and the Phase 1 design note, phase report, and
  ADRs 0001 (module architecture) and 0002 (math and depth conventions).
- `cyber_core` reference module with version metadata and a CTest smoke test.
- Install and CMake package export: `find_package(CyberGraphics CONFIG)` with
  `cyber::core` and `cyber::graphics_api` targets.
- `tests/integration/package_consumer` verifies the installed package on
  Windows and Linux CI.
- Single-source version generation from `project(VERSION)`.
- Explicit MSVC runtime selection (`/MD` by default, `/MT` option).
- Formatting gate (`scripts/check_format.py`), SHA-pinned GitHub Actions, and
  Dependabot updates.
- Integration guide, ADR 0003, and `SECURITY.md`.
- Phase 2 `cyber_core` primitives: fixed-width types, `BuildConfig`, `Error`,
  `Result<T, E>`, assertions, a bounded logger with sinks, and xxHash64.
- Project test runner in `tools/test` with `CG_TEST`, `CG_EXPECT_*`,
  `CG_REQUIRE`, listing, filtering, repeat, seed, shuffle, and console/TAP/JUnit
  output.
- ADR 0004, the Phase 2 design note, and the Phase 2 report.
- Phase 3 `cyber_math`: vectors, column-major matrices, quaternions, TRS
  authoring, right-handed reverse-Z view/projection matrices, AABB/plane/ray/
  sphere/frustum geometry, linear/sRGB color, and IEEE binary16 storage.
- Compile-time SSE2 `Vec4` fast path with a scalar fallback and SIMD-versus-
  scalar benchmark (`cyber_math_bench`).
- Absolute-tolerance test assertions: `CG_EXPECT_NEAR` and `CG_REQUIRE_NEAR`.
- ADR 0005, the Phase 3 design note, and the Phase 3 report.
- Phase 4 `cyber_memory`: the `Allocator` interface, `HeapAllocator`, linear,
  stack, pool, free-list, ring, frame, and tracking allocators, plus tagged leak
  reports and allocator accounting.
- `Span`, `StringView`, `String`, `DynamicArray`, `FixedArray`, open-addressing
  `HashMap`, `Handle<Tag>`, and generation-checked `SlotMap<T, Tag>`.
- Deterministic 128-block free-list stress coverage, deliberate-leak coverage,
  and stale-handle coverage in the memory tests.
- `cyber::memory` install/export and source/package consumer coverage.
- ADR 0006, the Phase 4 design note, and the Phase 4 report.
- UBSan builds exclude the `vptr` check because runtime code and tests share
  RTTI-disabled settings; test targets keep exceptions enabled.
- Phase 5 `cyber_platform`: renderer-agnostic windows, fixed-size event queues,
  keyboard/mouse input, native high-resolution clocks, frame timer,
  fixed-timestep helper, UTF-8 file I/O, read-only memory mapping, path helpers,
  polling file watching, native threads, synchronization wrappers, thread
  naming, and dynamic-library loading. Win32 lives under `src/win32/`; other
  platforms get a null window backend.
- Phase 5 `cyber_perf`: non-allocating `Profiler` interface, `NullProfiler`, and
  a process-lifetime null instance.
- Phase 5 `cyber_jobs`: fixed worker pool, bounded per-worker work-stealing
  deques, dependency counters, generation-checked handles, `SubmitDetached`,
  `Wait`, `WaitIdle`, `ParallelFor`, profiler scope markers, and deterministic
  shutdown.
- `samples/00_window` proves the platform layer runs without a renderer.
- `cyber_jobs_bench` measures 10,000 jobs at 1, 2, 4, and hardware worker
  counts and reports wall time, speedup, and throughput.
- `cyber::platform`, `cyber::perf`, and `cyber::jobs` install/export and are
  exercised by the source and package consumers.
- ADR 0007, the Phase 5 design note, and the Phase 5 report.
- MSVC builds define `_HAS_EXCEPTIONS=0` for every target so standard-library
  exception ABIs agree under shipping LTO; tests keep `/EHsc` unwinding.
- Phase 6 `cyber_software_renderer`: linear `RGBA8Unorm`/`RGBA32F` framebuffer
  with clipped fills, pixel access, blits, and format conversion; sRGB present
  packing through a 16 KiB lookup table (within one 8-bit code of
  `EncodeSrgb8`); deterministic PPM and TGA export; a 5x7 bitmap font; and a
  60 Hz `FramePacer` with catch-up clamping, frame statistics, and a bounded
  1.5 ms spin that cancels OS timer rounding.
- Phase 6 `cyber_platform`: allocation-free `Presenter` seam over caller-owned
  RGBA8 bytes with a Win32 `StretchDIBits` backend (BGRA swizzle,
  allocator-backed scratch, fixed-point nearest sampling), plus
  `SleepUntilSeconds`/`SleepSeconds` built on a per-call high-resolution
  waitable timer on Windows and `nanosleep` on POSIX.
- `samples/01_gradient` shows an animated linear gradient at 60 Hz with a
  frame-time overlay, `F2`/`F11`/`Escape` controls, a `--frames` smoke run that
  prints the per-frame cost breakdown, and a `--capture` mode that proves two
  images of the same frame are byte-identical.
- `cyber_software_renderer` installs/exports and is exercised by the source and
  package consumers.
- ADR 0008, the Phase 6 design note, and the Phase 6 report.
- Phase 7 `cyber_software_renderer`: clipped all-octant Bresenham lines;
  fixed-point edge-function triangle fill with winding normalization, the
  clockwise top-left rule, and unbiased barycentric vertex colors; tile binning
  with one `cyber_jobs` job per tile, producing bytes independent of worker
  count and tile size; guard-band, degenerate, and non-finite rejection; and
  `RasterStats` with per-tile pixel-write counts.
- A 10,000-triangle jittered mesh covers 640x360 exactly once, with serial and
  parallel output byte-identical across tile sizes 16/64/128 and one/four
  workers.
- `samples/02_raster` shows the animated triangle mosaic at 60 Hz with a
  frame-time overlay, a `--frames` smoke run, and a `--capture` mode that proves
  two images of the same frame are byte-identical.
- ADR 0009, the Phase 7 design note, and the Phase 7 report.
- Phase 8 `cyber_software_renderer`: `DepthBuffer` (float reverse-Z, clear 0,
  greater-or-equal test), `ClipTriangleNearPlane` and homogeneous near-plane
  clipping, `DrawTriangles` (model/view/projection, viewport y flip, back-face
  culling, +/-16384-pixel guard band), perspective-correct color through
  `1 / w`, `pixelsRejected` depth statistics, and the `Camera`, `FlyCamera`, and
  `OrbitCamera` values and controllers.
- Full-frame clears now fill rows by doubling a pixel pattern, and the depth
  clear uses `memset` for the all-zero reverse-Z far value; the Phase 7 capture
  hash is unchanged.
- `samples/03_depth` renders a depth-tested rotating cube over a perspective
  checkerboard floor with orbit/fly cameras, holds 60.0 FPS at 1280x720 in
  release, and captures byte-identical fixed frames including an inside-the-cube
  near-clipping pose.
- ADR 0010, the Phase 8 design note, and the Phase 8 report.
- Phase 9 `cyber_software_renderer`: a move-only `Texture` with a box-filtered
  mip chain, repeat/clamp/mirrored-repeat addressing, and nearest, bilinear, and
  trilinear sampling; a CPU shader interface (`VertexShaderFn`/`PixelShaderFn`
  over `VertexInput` and `Varyings`); `DrawShadedTriangles` and
  `RasterizeShadedTriangles` with perspective-correct varying interpolation and
  near-plane clipping; Lambert and Blinn-Phong lighting with directional and
  point lights; PPM (P6/P5/P3) and TGA (24/32-bit, both origins) readers that
  decode sRGB to linear; an OBJ reader with fan triangulation and negative
  indices; and a golden-image harness with per-channel tolerance and a
  difference image.
- Five committed 640x360 golden scenes (`checker_cube`, `lambert_cube`,
  `blinn_cube`, `uv_gradient`, `depth_order`) compared in every preset, plus a
  determinism check that renders twice byte-identically.
- `samples/04_textured` renders a checker-textured, lit cube with a Blinn-Phong
  highlight and captures byte-identical fixed frames. Its `--supersample` flag
  renders `n` samples per axis and box-filters the result in linear space, which
  antialiases the silhouette until the rasterizer has multisample coverage; the
  window lowers the factor to stay within 2560x1440.
- CI now runs AddressSanitizer on Linux through the `ci-linux-asan` preset.
- ADR 0011, ADR 0012 (supersampling until multisample coverage exists), the
  Phase 9 design note, `docs/testing-golden-images.md`, and the Phase 9 report.
- Phase 10 `cyber_rhi`: the GPU object model. Eleven generation-counted handle
  types; a `Format` enum with depth, sRGB, block-compressed, and integer
  classification; `BackendKind`, `AdapterInfo`, `Limits`, `Feature`, and
  `Capabilities`; the boundary conventions (+Y up, reverse-Z in 0..1, top-left
  texture origin) with a per-backend mapping table; a function-pointer backend
  boundary (`Backend`, `DeviceInterface`); a move-only `Device` with one queue
  per class, monotonic 64-bit timeline fences, and `WaitIdle`; and the null
  backend that validates every call.
- ADR 0013 (RHI object model), the Phase 10 design note, the D3D12/Vulkan/WebGPU
  comparison notes, and the Phase 10 report.

### Changed

- Temporal anti-aliasing no longer weights contributions by `1 / (1 + luma)`
  by default. The weighting suppresses a stray bright sample, and it biases the
  accumulation darker wherever a pixel alternates between a bright and a dark
  value — which is what a sub-pixel highlight under jitter does. On the
  acceptance scene it cost roughly an order of magnitude of converged accuracy.
  It remains available through `TemporalAaSettings::luminanceWeighting` for
  scene-referred HDR input.

### Fixed

- The D3D12 backend set a triangle-list topology on every pipeline bind,
  whatever the pipeline declared. A pipeline state fixes only the topology
  type, so point and line pipelines drew nothing. The pipeline now records its
  topology and the list sets it when the pipeline is bound.
- A `ShaderWrite`-to-`ShaderWrite` buffer barrier recorded nothing on D3D12, so
  two dispatches writing the same buffer were not ordered. It now records a UAV
  barrier.
- A `ShaderRead` transition on a D3D12 compute command list named the
  pixel-shader-resource state, which compute lists may not use, and `Close`
  failed. On a compute list the transition now names only the non-pixel state.

- The software pipeline dropped any triangle whose projected position left the
  rasterizer's +/-16384-pixel guard band, counting it as off screen. Ordinary
  geometry reaches that bound: a ground plane under the camera has corners
  behind the eye, and the near clip projects them far outside the band, so
  whichever triangle carried one disappeared and took a wedge of the scene
  with it. All three draw paths now clip to the guard band in homogeneous clip
  space before the perspective divide, which is exact, and the bound is left in
  place as a check rather than a reject. `software_pipeline_ground_plane_under_the_camera_is_whole`
  renders a plane under the camera and asserts the frame has no hole in it.

Found by rendering the two reference images and by the debug layer's own
messages, which the backend now reports on the `Rhi` channel.

- Direct3D 12: `SetPipeline` bound the root signature but never set the pipeline
  state, so every draw rasterized nothing while the frame recorded, submitted,
  and cleared correctly. `d3d12_commands_draw_a_colored_triangle` reads the
  uploaded bytes, the clear, and the shaded pixel back to catch it.
- Direct3D 12: a bind group read its descriptors back out of a shader-visible
  heap, which the API forbids because that heap is write-only for the CPU. Views
  and samplers are now built straight into the group's own descriptor.
- Direct3D 12: a color target advertised an optimized clear value that disagreed
  with the clear color every pass actually used; color targets now carry none,
  and depth targets carry the reverse-Z far plane the pass clears to.
- Direct3D 12: a barrier batch whose every entry was already in its target state
  still called `ResourceBarrier` with no barriers. An entry already in the
  tracked state now records nothing, and an empty batch makes no call.
- `cyber_rhi`: a submitted command list released before its fence was drained
  released the allocator the queue was still reading from; `Device`'s
  `CreateCommandList` now states the rule, and the test that broke it drains
  first.

### Notes

- The public API, renderer, and tools are not implemented yet. Planned modules
  intentionally have folders and contracts but no CMake targets.
- `cyber_rhi` implements its object model only: resource, pipeline, and
  command-list creation arrive in Phases 11 to 13.
- The project test runner is a developer tool; it is not installed and is never
  linked into a shipping target.
- `Transform` is an authoring record. Non-uniform TRS is not closed under
  composition, so hierarchies compose `ToMatrix()` values and invert through
  `TryInverse(Mat4&)`.
