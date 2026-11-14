# Phase 32 design — GPU-driven rendering and compute

Date: 2026-11-13
Status: Complete for Direct3D 12 on this machine's adapters. GPU instance
culling with LOD selection and Hi-Z occlusion, ordered draw compaction, and
indirect draws with a GPU-written count run on the HD Graphics 530 and on
WARP, checked against a CPU reference that is itself checked against Phase 22.
The mesh-shader pipeline is not built: meshlets and their culling are, and the
capability gate falls back to the vertex pipeline.
Related: [Phase 32 roadmap](../graphics/CYBERGRAPHICS_ROADMAP.md#phase-32-gpu-driven-rendering-and-compute),
[Phase 32 report](../reports/phase-32.md),
[Phase 22 design](phase-22-scene-culling.md),
[Phase 23 design](phase-23-geometry.md),
[Phase 27 design](phase-27-lighting.md),
[Phase 31 design](phase-31-gi-raytracing.md)

## Scope and boundaries

The roadmap asks for GPU-side culling, indirect multi-draw, mesh shaders where
supported, and async compute, and is done when a million objects are culled on
the GPU and drawn with a handful of calls while CPU time stays flat as the
object count grows. The build guide adds that the results match the CPU culler.

Unlike Phases 28 to 31, this phase's main path runs on the GPU. The development
adapter, an Intel HD Graphics 530, supports compute, `ExecuteIndirect` with a
count buffer, and a compute queue; everything the done-when line names can be
run and measured here.

In scope:

- RHI: real `DrawIndirect`, `DrawIndexedIndirect`, and `DispatchIndirect` on
  D3D12 through command signatures, and new `DrawIndirectCount` /
  `DrawIndexedIndirectCount` calls behind `Feature::IndirectDrawCount`, on
  the null, validating, and D3D12 backends.
- RHI: `Feature::MeshShaders` reported from the driver's mesh-shader tier.
- `gpu_driven.h`: the scene description, the shared GPU data layouts, the Hi-Z
  pyramid and occlusion test, a CPU reference culler that runs the culling
  shader's arithmetic, and the geometry-path and async-compute policies.
- `gpu_culling_pipeline.h`: the GPU side — three compute pipelines (Hi-Z
  reduction, instance culling, draw compaction) and the commands that run them
  on a graphics or compute queue.
- `gpu_light_culling.h`: cluster light lists built by a compute shader, so
  light culling can share the compute queue with instance culling.
- `meshlet.h`: meshlet building, sphere and normal-cone culling (the
  amplification stage's work), and the vertex-pipeline fallback.
- `samples/17_gpu_driven`: a million-instance landscape drawn this way.

Out of scope, and stated rather than implied:

- **The mesh-shader pipeline.** Mesh and amplification shaders need shader
  model 6.5, which only DXC compiles; the RHI compiles HLSL with `D3DCompile`,
  whose ceiling is 5.1 (ADR 0019). The development adapter reports no
  mesh-shader tier either, so a pipeline built now could not be run here.
  `kMeshShaderPathCompiledIn` is `false`, and a capable adapter resolves to the
  vertex pipeline with the reason `NotImplemented`, as Phase 31's ray-tracing
  gate does.
- **Two-phase occlusion.** Culling tests against the previous frame's depth
  only; see [Hi-Z occlusion](#hi-z-occlusion).
- **Render-graph integration.** The pipeline records into command lists the
  caller owns and tracks its own buffers' states; the graph does not schedule
  it yet.
- **Shading with GPU light lists.** Light culling builds the lists; a forward
  pass that reads them is Phase 27's outstanding GPU work.

## RHI changes

### Indirect execution

Phase 23 defined the indirect argument layouts and the null backend accepted
indirect draws, but the D3D12 backend returned `Unsupported` for every indirect
call. It now records `ExecuteIndirect`:

- A command signature fixes one argument type and one byte stride. The device
  creates each `(type, stride)` signature on first use and keeps it for its
  lifetime in a fixed 16-entry table, so the record path does not allocate.
  None of them changes a root argument, so none needs a root signature and
  every pipeline can share them.
- Both buffers are resolved under the device lock and range-checked against
  the CPU-side maximum, which is the most the device can read whatever the
  GPU-written count says. The last argument structure only needs its own
  bytes, not a whole stride.
- `IndirectCountArguments` names the argument buffer, the count buffer, their
  offsets, the maximum, and the stride; zero means tightly packed. The call
  counts as one draw in `CommandListStats` whatever the GPU decides.

`Feature::IndirectDraw`, `MultiDrawIndirect`, and `IndirectDrawCount` are
reported unconditionally on D3D12: `ExecuteIndirect`, including the count
buffer, is core on every feature level. On Vulkan the count variant is a 1.2
feature, which is why it has its own bit.

### Barriers between compute passes

Two dispatches that write one buffer need a UAV barrier between them, and the
backend dropped any transition whose two states were equal. A
`ShaderWrite`-to-`ShaderWrite` barrier now records a UAV barrier. That is what
orders instance culling before compaction, and each Hi-Z level before the
next.

### Compute lists and graphics-only states

The RHI's `ShaderRead` maps to `PIXEL_SHADER_RESOURCE |
NON_PIXEL_SHADER_RESOURCE`, and a compute command list may not name the pixel
half: `Close` fails with no message to say why. On a compute list the backend
now strips that bit, so a shader read there is a non-pixel read.

### Primitive topology

`SetPipeline` always set a triangle-list topology. A pipeline state fixes only
the topology type, so a point or line pipeline was fed triangles and drew
nothing. The pipeline now records its topology and the list sets it on bind.
This predates the phase; the point pipeline the acceptance tests draw with is
what found it.

### Mesh-shader capability

`Feature::MeshShaders` is reported from `D3D12_FEATURE_D3D12_OPTIONS7` at
`D3D12_MESH_SHADER_TIER_1` or above, never inferred. A failed query reports
nothing.

## Data model

```text
GpuDrivenScene
  groups   [g]  lodCount, firstBatch, instanceCount, LOD thresholds
  batches  [b]  indexCount, firstIndex, vertexOffset, visibleBase, capacity
  instances[i]  world bounds, drawGroup, instanceId
visible list    batch b owns [visibleBase, visibleBase + capacity)
```

A **draw group** is a mesh with up to four LODs, using Phase 22's
screen-space thresholds. Each LOD is a **batch**: one indexed draw. Any
instance may select any LOD of its group, so every batch of a group gets a
capacity equal to the group's instance count, and the visible list holds the
sum of all capacities. That costs memory — four bytes per instance per LOD —
and buys a culling pass with no global prefix sum: an instance appends to its
batch's slice with one atomic.

Each batch's draw takes `firstInstance = visibleBase`. A per-instance vertex
attribute is offset by `firstInstance`, so binding the visible list as an
`R32Uint` stream at `kGpuDrivenInstanceStreamBinding` hands every draw exactly
its batch's identifiers; `SV_InstanceID` would not, because D3D12 does not add
`StartInstanceLocation` to it.

## The frame

```text
            compute or graphics queue                       graphics queue
 ┌─────────────────────────────────────────────┐   fence  ┌────────────────────┐
 │ clear counts ─► cull (N/64 groups) ─UAV─►   │ ───────► │ acquire ─► one     │
 │                compact (1 group)            │          │ DrawIndexedIndirect│
 │ [Hi-Z reduction ─UAV─► ... before cull]     │          │ Count              │
 └─────────────────────────────────────────────┘          └────────────────────┘
```

1. **Clear.** Batch counts and the counter block are copied from a zero
   buffer written once at creation; the cull constants come from one of three
   staging slots, so three frames may be in flight.
2. **Cull.** One thread per instance, 64 per group; past 2,097,152 instances
   the dispatch spills into Y. Frustum test, LOD selection, optional Hi-Z test,
   then an atomic append into the batch's slice. Statistics are reduced in
   group-shared memory first, so a million instances cost one global atomic per
   counter per group, not per instance.
3. **Compact.** One group of 256 threads. Each thread owns a contiguous run of
   batches and counts its non-empty ones; a Hillis-Steele scan turns the
   counts into write positions, and the draws come out in batch order. The
   draw count is the scan's total. The order is deterministic and equal to the
   CPU reference's, not merely equivalent.
4. **Draw.** The arguments and count move to `IndirectArgument`, the visible
   list to `VertexBuffer`, and one `DrawIndexedIndirectCount` with
   `maxDrawCount = batch count` draws every surviving batch.

The CPU's share is fixed: a 224-byte constant block and about twenty commands,
whatever the instance count.

## Culling arithmetic, and agreeing with the CPU

`CullInstancesReference` is the oracle: the culling shader's decisions on the
CPU, in the same operation order. Its frustum test is Phase 22's
`Intersects` — the planes from `Frustum::FromViewProjection`, the positive
vertex per plane, rejection at the first negative distance — and its LOD rule
is Phase 22's, so the reference agrees with `Scene::Cull` exactly, and the
suite asserts it.

The shader marks every expression that decides visibility `precise`, which
stops the compiler fusing a multiply-add or reassociating a sum. Additions,
multiplications, and comparisons then round the same on both sides. Division
and square root do not: D3D allows them a few ULPs. So
`DecideGpuInstance` reports, beside each decision, how close it came to the
other answer — the smallest plane distance, the relative distance between the
projected radius and a LOD threshold, and for occlusion the relative depth
margin and the pixel margin to a texel boundary. A test of the GPU path calls
a disagreement *explained* only if one of those margins is within rounding,
and fails on any other. On both local adapters the measured count of
disagreements, explained or not, is zero.

## Hi-Z occlusion

**Layout.** Level 0 is the depth buffer; level `k` is `ceil(level(k-1) / 2)`
per axis, down to 1 x 1, all levels packed in one buffer. With ceiling sizes
the parent of texel `i` is `i >> 1` at every level, so the reduction needs no
remainder rule: a texel's children are `2x` and `2x + 1`, the second clamped.
Each texel holds the farthest depth below it — the minimum, in reverse-Z.

**Build.** `RecordHiZBuild` copies level 0 from a depth buffer in GPU memory
(with a row pitch, so a texture copied to a buffer works directly) and reduces
each level from the one below, one dispatch per level, with a UAV barrier
between. `RecordHiZUpload` takes a pyramid the CPU built instead. A reduction
of minima has nothing to round, and the GPU pyramid equals the CPU one texel
for texel.

**Test.** The box's eight corners are projected with the view-projection the
pyramid was rendered with. A corner at or behind the camera, or a box entirely
outside that view, makes the box untestable, and it is kept. Otherwise the
nearest depth of the box is the largest corner NDC depth — the projected depth
is linear-fractional over the box, so its extremes are at corners — and the
screen rectangle's texel range is floored and clamped. The level read is the
finest at which that range spans at most two texels per axis, so four reads
cover it. The box is occluded only if its nearest depth is farther than the
farthest occluder in all four. A box called occluded is therefore behind the
depth buffer at every pixel its rectangle touches, and the suite checks that
against level 0 directly.

**Latency.** Culling against the previous frame's depth, reprojected with the
previous frame's view-projection, is conservative for static occluders but
lags by a frame: an object revealed by camera motion or a moving occluder
appears one frame late. Two-phase occlusion — re-testing last frame's culled
set against this frame's depth — removes the lag and is the next step.

**Level selection on WARP.** The first shader computed the level as the
smallest `l` with `(x1 >> l) - (x0 >> l) <= 1`, the CPU's form. On WARP, 1,688
of 200,000 boxes read a coarser level than that condition allows, though the
rectangle, the pyramid, and the depth were all identical to the CPU's; the
HD Graphics 530 agreed exactly. The shader now steps the coordinates down one
level at a time with constant one-bit shifts, which is the same function, and
both adapters agree with the CPU on every instance. The fault inside WARP was
not isolated further.

## Meshlets and the mesh-shader path

`BuildMeshlets` splits an indexed triangle list in index order, closing a
meshlet when the next triangle would exceed 64 vertices or 124 triangles —
the sizes the D3D12 guidance gives, 124 keeping the primitive block in 512
bytes. Each meshlet stores a vertex list, one-byte local triangle indices, a
bounding sphere, and a normal cone.

`CullMeshlet` is the amplification stage's test: the sphere against each
plane, then `dot(c - e, a) >= s·|c - e| + r` for center `c`, eye `e`, cone
axis `a`, radius `r`, and `s` the sine of the cone's half-angle. It holds only
when every point of the sphere sees every normal in the cone from behind. The
cone is widened by 10^-3 in cosine to absorb rounding, and a spread wider than
about 84 degrees disables it. The suite checks both tests by brute force: every
triangle of a back-face-culled meshlet faces away from the eye at all three
vertices, and every triangle of a frustum-culled one is outside a plane.

`ExpandMeshletsToIndices` is the fallback: survivors become an ordinary index
list for the vertex pipeline, so the culling still removes work where mesh
shaders are absent. `SelectGeometryPath` resolves the preference: vertex only,
or mesh shaders where the adapter reports them and the build contains them.

## Async compute

`SelectAsyncComputePlan` puts instance culling and light culling on the
compute queue only when:

1. the preference is not `Off`;
2. the device has a compute queue; and
3. under `Auto`, a measurement of at least eight frames each way shows the
   async schedule saving at least 5% of the serial frame. Without a
   measurement `Auto` stays serial. `Force` exists to take the measurement.

The schedule itself is plain: the compute submission signals a fence and the
graphics submission waits for it with an `rhi::GpuWait`, the only cross-queue
ordering the RHI has. The pipeline tracks its buffers' states in recording
order, so lists are submitted in the order they were recorded. The draw-side
transitions happen on the graphics list, after the wait.

On the HD Graphics 530 the measured difference is within noise (−1.1% and
+0.5% in two runs), so `Auto` declines — the build guide's warning about this
GPU, measured.

## Light culling

`GpuLightCullingPass` builds Phase 27's cluster lists in a compute shader,
one thread per cluster, using `BuildClusterLightLists`' projection. The GPU's
logarithmic slice range is widened by one slice each way so that its
single-precision logarithm can never drop a light the CPU's double-precision
one keeps: the lists are conservative supersets of the CPU's, and the CPU
stays the tight reference. It records on either queue; the acceptance test runs
it in the same compute submission as instance culling.

## Measurement

| Property | Test | Asserted against |
|---|---|---|
| GPU culling equals the CPU | `gpu_driven_d3d12_culls_a_million_instances_like_the_cpu` | Reference, per instance; Phase 22 in total and per LOD |
| Drawn equals culled | `gpu_driven_d3d12_draws_a_culled_million_with_one_call` | One lit texel per surviving identifier |
| GPU pyramid equals CPU | `gpu_driven_d3d12_hiz_build_matches_the_cpu_pyramid` | Bitwise |
| GPU occlusion equals CPU | `gpu_driven_d3d12_occlusion_culling_matches_the_reference` | Reference, per instance |
| Occlusion is conservative | `gpu_driven_hiz_occlusion_is_conservative` | Level-0 depth under every covered pixel |
| Async equals serial | `gpu_driven_d3d12_async_compute_culling_matches_the_graphics_queue` | The single-queue image |
| CPU time is flat | `gpu_driven_d3d12_cpu_time_stays_flat_as_objects_grow` | 1k vs 1M instances; a CPU-driven control that grows |
| Light lists are complete | `gpu_driven_d3d12_light_and_instance_culling_share_the_compute_queue` | `BuildClusterLightLists` |

Each threshold has a control that fails it: culling with occlusion off draws
the hidden block; the CPU-driven control's time grows forty-fold over a
hundredfold increase in instances.

## Determinism and threading

The reference culler decides instances in parallel chunks and appends them in
instance order, so its output is byte-identical with and without a job
system. The GPU's visible lists are appended with atomics, so the order inside
one batch is unspecified there; tests compare sorted slices. Draw compaction
is ordered, so the GPU's draws equal the reference's byte for byte.

`GpuDrivenScene`, `HiZPyramid`, `GpuCullingPipeline`, and
`GpuLightCullingPass` are not thread-safe. Query functions are pure.

## Known limitations

- **No mesh-shader pipeline**, for the reasons in the scope.
- **One-frame occlusion latency**, without two-phase culling.
- **Visible-list memory** is four bytes per instance per LOD of its group.
- **The pipeline is outside the render graph**, and tracks its own states.
- **Buffer state across submissions.** The D3D12 backend tracks states at
  record time and does not model D3D12's decay of buffers to `COMMON` at the
  end of each `ExecuteCommandLists`. The debug layer accepts the resulting
  barriers, because a buffer in `COMMON` is promoted to any state; the backend
  makes no stronger claim than that.
- **Light lists are built, not consumed.** The GPU forward pass that shades
  with them is Phase 27's outstanding work.
