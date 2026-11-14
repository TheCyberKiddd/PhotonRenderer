# Phase 27 design — clustered lighting

Date: 2026-11-08
Status: CPU clustered-forward path complete; standalone D3D12 compute proof validated
Related: [Phase 27 roadmap](../graphics/CYBERGRAPHICS_ROADMAP.md#phase-27-clustered-lighting-and-deferred-option), [build guide](../graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-27-clustered-lighting-and-deferred-option), [renderer module contract](../../engines/cyber_renderer/docs/MODULE.md)

## What this phase builds

This phase delivers a CPU clustered-forward path and a standalone D3D12 compute
proof. The CPU builds per-view cluster lists and the software rasterizer
consumes them while shading PBR fragments. The D3D12 sample separately builds
bounded cluster lists and shades synthetic surface records with a GGX compute
shader, then validates list membership and representative pixels against the
CPU implementations.

The deterministic builder, occupancy data for heat-map visualization,
clustered PBR consumer, differential tests, and an end-to-end 1,000-light CPU
frame benchmark are implemented. The D3D12 compute sample is a headless proof,
not renderer-plugin integration: it uses synthetic surfaces rather than
geometry rasterization or a G-buffer, and it does not select or fall back
between renderer backends. See the [implementation report](../reports/phase-27.md)
for validation evidence and limitations.

## What it deliberately leaves out

This phase does not change the renderer plugin ABI, add a new RHI backend, or
make deferred rendering the default. The compute proof is D3D12-only and does
not probe capabilities or switch an existing renderer to the CPU path.
Directional lights remain a small global list. Transparent draw ordering and
integration remain the responsibility of the existing renderer path. A
rendered heat-map overlay and deferred comparison remain future work.

## Ownership and data flow

- `cyber_renderer` owns the backend-neutral cluster grid and deterministic CPU
  list builder. `cyber_software_renderer` owns CPU light-bound conversion and
  clustered PBR shading; the renderer module does not depend on the software
  renderer.
- The CPU path is selected by using the software renderer and supplying its
  CPU-built lists. It is not an implicit fallback inside a GPU renderer.
- The standalone D3D12 sample owns its handwritten HLSL and explicit RHI
  layouts. This is intentionally isolated because the shader DSL does not yet
  express the structured-buffer loops used by the proof; it does not alter the
  renderer module's shader-reflection contract.
- GPU surface records stand in for rasterized or G-buffer data. Deferred
  rendering and a rendered heat-map overlay are not delivered.

## Cluster and light-list rules

- Divide the viewport into 16-pixel XY tiles and the camera's positive
  view-space depth into 24 logarithmic slices between its validated near and
  far planes. Keep these initial dimensions fixed for the benchmark so runs
  are comparable; revisit them only with measured evidence.
- Construct each cluster's view-space frustum from the camera projection.
  Point lights intersect clusters by their finite range spheres. Spot lights
  initially use the same conservative range sphere; this may add lights but
  must not omit contributing lights. Directional lights bypass the cluster
  lists.
- Store indices into one frame-local light array, with checked offsets and
  counts. Reject non-finite or invalid ranges and report arithmetic and
  allocation failures explicitly. Never silently truncate a cluster's list.
- The CPU builder uses exact-sized lists. The compute proof uses a fixed
  capacity of 128 indices per cluster, records overflow explicitly, and falls
  back to testing every local light for shading when a list overflows.
- Cluster lookup derives XY from the pixel and Z from the interpolated
  world-position transformed into view space. Depth boundaries, near/far
  clipping, viewport edges, and empty clusters have deterministic ownership.

## Deferred and transparency policy

Clustered forward reuses the established PBR material path. The
software-renderer path has no GPU transparency sorting or draw integration;
callers remain responsible for their renderer's existing transparent ordering. A
deferred/G-buffer experiment is deferred until a GPU forward path exists for a
meaningful comparison.

## Verification and acceptance

- CPU unit tests cover cluster addressing, depth boundaries, conservative
  light assignment and projection, invalid input, allocation failures, empty
  and edge clusters, deterministic output, exact list contents, heat-map data,
  and reuse of allocated storage.
- The software-PBR tests compare 1,000-light output against brute force and
  exercise the rasterizer across representative pixels and fallback cases.
- The reproducible CPU frame benchmark reports resolution, processor/worker
  count, configuration, median and p95 frame time, occupancy, index count, and
  rendered-pixel coverage. Performance is recorded as a hardware-specific
  baseline; it is not represented as a 60-fps or GPU performance result.
- The D3D12 sample checks that every CPU-conservative light appears in the
  corresponding GPU list, rejects any overflow for its deterministic test
  scene, and compares 12 representative compute-shaded pixels against
  `PbrPixelShader` with a maximum absolute RGB error of `2e-3`.
- The sample is validated on the current D3D12 adapter only; this is not a
  cross-vendor hardware matrix, a full rasterized-scene test, or a performance
  claim.

## Risks

Conservative cluster bounds can create large lists near the camera or for
large-radius lights. CPU output uses checked exact-sized lists; the compute
sample's bounded lists use an explicit brute-force overflow fallback. The
software renderer is slower than a dedicated GPU path and its measured frame
time must not be presented as a 60-fps guarantee. Production GPU integration
must preserve the shader-reflection contract and validate its output against
the CPU builder before it is selected.

## Sources

- Ola Olsson, Markus Billeter, and Ulf Assarsson, [Clustered Deferred and Forward Shading](https://www.cse.chalmers.se/~uffe/clustered_shading_preprint.pdf).
- J. J. G. van Waveren, [Tiled and Clustered Forward Shading: Supporting Transparency and MSAA](https://jcgt.org/published/0002/02/03/).
- [Phase 27 implementation checklist](../graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-27-clustered-lighting-and-deferred-option).
