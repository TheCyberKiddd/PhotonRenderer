# Phase 31 design — global illumination and ray tracing

Date: 2026-11-13
Status: Complete for the software-reference scope: the CPU ray tracer, the
baked irradiance volume and its renderer integration, the spatio-temporal
denoiser, the GI metric API, the measured acceptance suite, and the
capability-gated ray-tracing selection policy are implemented. GPU ray tracing
is not.
Related: [Phase 31 roadmap](../graphics/CYBERGRAPHICS_ROADMAP.md#phase-31-global-illumination-and-ray-tracing),
[Phase 31 report](../reports/phase-31.md),
[ADR 0028](../adr/0028-gi-approach.md),
[Phase 29 design](phase-29-ibl-reflections.md),
[Phase 30 design](phase-30-post-processing.md)

## Scope and boundaries

The roadmap asks for one GI method shipped first and an optional hardware
ray-tracing path that switches on only where the GPU supports it, with a
denoiser. [ADR 0028](../adr/0028-gi-approach.md) records the choice: baked
irradiance volumes, chosen over lightmaps, screen-space GI, and voxel GI. The
build guide names the ADR `0003-gi-approach.md`; ADRs in this repository are
numbered in the order they are written, and the next free number was 0028.

In scope:

- A CPU ray tracer with DXR's structure: bottom-level and top-level
  acceleration structures, closest-hit and any-hit queries, instance masks.
- A Monte Carlo path tracer for diffuse light transport over that tracer,
  used both as the baker and as the ground truth the bake is judged against.
- An irradiance volume: a regular grid of order-2 spherical-harmonic probes,
  each with an octahedral map of visibility-distance moments, baked by the
  path tracer and sampled with normal and Chebyshev visibility weighting.
- Integration of the volume into `PbrPixelShader` as the diffuse ambient term.
- A spatio-temporal variance-guided denoiser for one-sample ray-traced
  signals.
- `gi_quality.h`, a measurement API, and an acceptance suite over it.
- `rhi::Feature::RayTracing` and `rhi::Feature::RayQuery`, reported by the
  D3D12 backend from `D3D12_FEATURE_D3D12_OPTIONS5`.
- `cyber::renderer::SelectRayTracingPlan`, which maps a policy and reported
  capabilities onto the technique that actually runs, per effect, with the
  rasterized path as the fallback.

Out of scope, and stated rather than implied:

- GPU acceleration-structure builds, ray dispatch, inline ray queries, and
  shader tables. The development adapter, an Intel HD Graphics 530, reports
  `D3D12_RAYTRACING_TIER_NOT_SUPPORTED`, and a GPU path that cannot be run here
  cannot be verified here. `kHardwareRayTracingCompiledIn` is `false`, and a
  capable adapter resolves to the rasterized path with the reason
  `NotImplemented` rather than to an absent implementation.
- The second-machine verification the build guide asks for. It needs a
  ray-tracing-capable GPU, which is not available.
- Lightmaps, incremental or runtime re-baking, and GPU-side volume sampling.
- Render-graph passes for the bake. The bake is offline CPU work with no GPU
  resource to schedule; it runs as an explicit call, not as a graph pass. The
  sampling side runs inside the forward pixel shader, which already executes
  within the software renderer's existing pass.

## The ray tracer

`raytrace.h` is the reference everything else in the phase is built on.

**Vocabulary.** A `BottomLevelAccelerationStructure` owns triangles in object
space. A `TopLevelAccelerationStructure` owns instances, each a borrowed
bottom-level structure plus a transform, an identifier, and an eight-bit mask.
A query is a `RayDesc` — origin, direction, `tMin`, `tMax`, as DXR spells it —
and is answered by `Intersect` (closest hit) or `IntersectAny` (first hit, for
visibility). `RayHit` reports the distance, position, geometric and shading
normals, barycentrics in DXR's convention, UV, primitive index, instance index,
instance identifier, and which face was hit. A hardware path replaces the
traversal behind the same shape, and this one stays as the oracle.

**Build.** Top-down binned surface-area heuristic: centroids are binned into
16 buckets along each axis, every bucket boundary is costed as the summed
child surface area weighted by primitive count, and the cheapest split is
taken if it beats testing every primitive in the node. A node that binning
cannot separate — every centroid in one bucket, which coincident geometry
produces — is split at its median centroid instead, ties broken by primitive
index so the order depends only on the input. The same builder orders
triangles and instances.

Three guarantees make the traversal simple:

- **Bounded depth.** A node at depth 63 is a leaf whatever its size, so a
  query uses a fixed 66-entry stack: no heap, no recursion.
- **Stable indices.** Degenerate triangles stay in the structure and are never
  hit. Dropping them would renumber every primitive after them.
- **Determinism.** Two builds of the same input produce identical node arrays,
  so a query visits the same nodes in the same order on every run.

Triangles are stored in leaf order as an origin and two edges, which is what
Möller–Trumbore consumes. The determinant test is relative to the edge and
direction lengths, so a small triangle is not mistaken for a degenerate one.

**Instances.** The top level inverts each transform once at build time and
keeps the inverse transpose for normals, so a non-uniform scale is handled
correctly; a singular transform is rejected at build rather than producing
silent misses. A query transforms the ray into object space without
renormalizing the direction, which keeps `t` identical in both spaces, so the
running closest distance carries across instances unchanged.

## Light transport

`TracePathRadiance` is a unidirectional path tracer for Lambertian surfaces.
At every interaction it gathers direct light from each directional and point
light with an any-hit shadow ray (next-event estimation), adds emission when
the path hits an emitter, and continues along a cosine-weighted direction, for
which the Lambertian estimator reduces to albedo times incoming radiance.
Emission and the sky are gathered only when hit, never light-sampled, so no
source is counted twice. Russian roulette terminates low-throughput paths
without bias after a configurable bounce.

Two conventions are worth stating:

- **Reflection is two-sided, emission is not.** A single-sided quad such as a
  floor lights correctly from either side, and an area light still aims.
- **Randomness is counter-based.** Each sample's stream is a hash of
  (sequence, sample) stepped by a counter. A bake is a pure function of its
  inputs, byte-identical with any number of workers, and the stream position
  never depends on a roulette outcome.

The white furnace is how this is checked. Inside a closed box whose walls emit
radiance 1 and reflect 0.5, every ray hits a wall, so with roulette off the
radiance after n bounces is exactly 1 + 0.5 + … + 0.5^n in every direction and
the estimator has no variance. The suite asserts it to rounding for n = 0..3,
and with roulette on, that the infinite series converges to 2.

## The irradiance volume

### Storage

A probe holds nine SH coefficients of incident radiance (27 floats), a 16 x 16
octahedral map of the mean and mean-square distance to the first surface (512
floats), the fraction of its rays that hit back faces, and a classification:
`Unbaked`, `Valid`, or `Inside`. That is 2,164 bytes per probe; the sample's
9 x 7 x 9 volume is 1.2 MB.

### Bake

Each probe traces `raysPerProbe` rays along a spherical Fibonacci lattice,
rotated per probe by a deterministic random rotation so that identical
lattices do not alias in the interpolated result. Each ray's radiance comes
from the full path tracer, so the stored SH includes every bounce the settings
allow, and convolving it with the cosine lobe at lookup adds one more.

A ray that hits a back face contributes zero radiance at its true distance:
what a probe sees through a back face is the inside of a solid. A probe whose
back-face fraction exceeds a threshold (0.25 by default) is classified
`Inside`, its SH is cleared, and sampling skips it. The block in the Cornell
room buries probes, and the suite asserts they are found.

Probes are baked into scratch storage and copied into the volume only once
every probe has finished, so a failed or invalid bake leaves the volume as it
was.

### Lookup

`SampleIrradianceVolume` pushes the point a quarter of the probe spacing along
the normal — testing visibility from exactly on a wall is ill-conditioned —
then weights the eight surrounding probes by:

1. **Trilinear position.**
2. **A smooth back-face term**, `((dot(toProbe, n) + 1) / 2)^2 + 0.2`, which
   down-weights probes behind the surface without the discontinuity a hard
   cut produces as a surface rotates past a probe.
3. **Chebyshev visibility.** The probe's distance moments are read bilinearly
   from its octahedral map in the direction of the point. If the point is
   farther than the mean, the one-tailed Chebyshev bound
   `variance / (variance + excess^2)` is cubed to sharpen it toward a step.
   Weights below 0.2 are crushed further so that several near-zero weights
   cannot add up to a visible contribution.

`Inside` and `Unbaked` probes are skipped and the weights renormalized. If
visibility rejects every valid probe, the lookup falls back to plain trilinear
over the valid probes rather than snapping to the fallback colour, which keeps
the answer continuous.

### Why the visibility map is 16 x 16

The resolution was measured, not chosen. The acceptance suite's leak test puts
probes 0.35 either side of a wall between a lit room and a sealed one. With
the visibility moments in six axis lobes, the lookup leaked 7.4% of the lit
side's irradiance on average and 35% at worst: the lobe toward the wall also
averaged the diagonal rays that run past it to the floor and ceiling, so the
test saw a wall that was farther away and fuzzier than it is. An 8 x 8
octahedral map cut the mean to 1.7% but still leaked 16% at the oblique angles
where its bilinear footprint spans 35 degrees and reaches a texel looking down
the room. At 16 x 16 — the resolution DDGI uses for the same moments — with a
cos^32 accumulation lobe, the mean leak is 0.04% and the worst 0.39%.

## Renderer integration

`PbrLightingData::irradianceVolume` points at an `IrradianceVolumeView`. When
set, the volume answers the diffuse ambient term:

- **Inside the image-based path** — a probe set or a global environment —
  `IblSurface::irradianceOverride` replaces only the environment's irradiance
  lookup. The split-sum specular term is unchanged, and the diffuse lobe still
  receives only the energy the single- and multiple-scattering specular lobes
  leave. Probes, screen-space reflections, and the environment all pick up
  baked GI through this one path.
- **Without image-based lighting** it replaces `ambientIrradiance`.

`ambientOcclusion` is applied in both cases. A probe grid cannot represent
contact-scale occlusion, and the screen-space ambient occlusion from Phase 30
is what supplies it.

The suite asserts the integration exactly: a volume reporting irradiance E
shades a pixel identically to `ambientIrradiance = E`, and an environment with
black diffuse SH plus the volume shades identically to an environment whose
diffuse SH reports E.

## Denoising

`SpatioTemporalDenoiser` follows Schied et al., *Spatiotemporal
Variance-Guided Filtering*:

1. **Temporal accumulation.** Each pixel is reprojected along the velocity
   buffer, and each of the four bilinear taps is tested against the previous
   frame's linear depth (10% relative) and normal (cosine 0.9). Surviving taps
   are renormalized; a pixel with none restarts. Colour and the first two
   luminance moments accumulate at `max(alpha, 1 / length)` — a true running
   mean while history is short, an exponential average once it is long.
2. **Variance.** From the temporal moments once history reaches four frames;
   before that, from a 5 x 5 bilateral spatial estimate, boosted for very short
   history as SVGF does.
3. **Edge-aware à-trous filtering.** Four passes of a 5 x 5 B3-spline kernel at
   steps 1, 2, 4, and 8. Each tap is weighted by depth, normal, and luminance
   similarity, the last scaled by the 3 x 3-prefiltered standard deviation, so
   the filter is strong where the signal is noisy and gentle where it has
   converged. Variance is filtered with squared weights so each pass refines
   the next pass's guide.

The first wavelet pass's output, not the raw accumulation, is fed back as
history, which converges faster and carries less noise forward. Guides follow
the post chain's conventions, so a frame's G-buffer feeds both without
conversion.

## Measurement

`gi_quality.h` turns the phase's properties into numbers, as `shadow_quality.h`
and `post_process_quality.h` did for theirs:

| Function | Measures | Asserted against |
|---|---|---|
| `MeasureIrradianceError` | Relative luminance error, signed bias, energy ratio | Path-traced ground truth at the same points |
| `MeasureLightLeak` | Dark-side over lit-side irradiance | A sealed room's ground truth, which is zero |
| `MeasureNoise` | Relative RMSE, signed bias, Laplacian energy | A converged reference |

Every threshold in the acceptance suite is paired with a configuration that
fails it: a bake with no interreflection, the same volume sampled without its
visibility and normal terms, and the raw one-sample signal.

## Ray-tracing capability gate

The RHI reports what the adapter can do, never what the build can use:

- `Feature::RayTracing` at `D3D12_RAYTRACING_TIER_1_0` or above.
- `Feature::RayQuery` at tier 1.1 or above, never without `RayTracing`.
- A failed `OPTIONS5` query reports neither. The null backend reports neither,
  because it has no ray-tracing objects to validate.

`SelectRayTracingPlan` resolves a `RayTracingPolicyDesc` against those
capabilities for each of four effects — shadows, reflections, global
illumination, ambient occlusion — each of which always has a rasterized answer
(Phase 28 shadow maps, Phase 29 SSR over probes, the Phase 31 volume, Phase 30
SSAO). The order of the checks is the policy:

1. Preference `Off`: rasterized, `PolicyDisabled`.
2. Effect not enabled: rasterized, `EffectDisabled`. Only shadows are enabled
   by default, which is the effect the build guide says to start with.
3. Preference `SoftwareReference`: the CPU tracer, whatever the adapter.
4. No `RayTracing`, or `RayQuery` without it: rasterized, `Unsupported`.
5. No GPU implementation in this build: rasterized, `NotImplemented`.
6. Otherwise inline queries where reported, else the ray-generation pipeline;
   `PreferPipeline` forces the latter.

The policy is pure, so it is tested against fabricated capability sets — the
way the build guide says to verify logic that the development GPU cannot run —
and against every D3D12 adapter the test machine can open. Both adapters here,
the HD Graphics 530 and the WARP software adapter, report no ray-tracing
support and resolve to the rasterized path.

## Determinism and threading

Every CPU path in the phase is deterministic. The BVH build depends only on
its input; the bake's random streams are index-derived and probes write
disjoint outputs; the denoiser's passes write disjoint rows. The suite asserts
byte-identical results with and without a four-worker job system for both the
bake and the denoiser. Queries are `const` and may run concurrently.

## Known limitations

- Indirect lighting is static. Moving geometry or lights invalidates the
  volume, and nothing re-bakes incrementally.
- Probe spacing bounds indirect detail; surfaces near `Inside` probes show
  mild variation at probe scale.
- The bake models every surface as Lambertian. Glossy interreflection is not
  captured.
- The octahedral lookup clamps rather than wraps at the map edge, which costs
  a little smoothness across the −Z fold.
- The denoiser has no history clamping against the current neighbourhood, so
  a lighting change under a still camera converges at the accumulation rate
  rather than snapping; a renderer driving it must call `Reset` on a cut.
