# 0028 — Global illumination approach and the ray-tracing capability gate

- **Status:** accepted
- **Date:** 2026-11-13
- **Deciders:** @TheCyberKiddd
- **Related:** Phase 31, [design note](../design/phase-31-gi-raytracing.md),
  [ADR 0027](0027-environment-map-ownership.md)

## Context

Phase 31 asks for one global-illumination method shipped first, plus an
optional hardware ray-tracing path that switches on only where the hardware
supports it. Each of the three candidates the roadmap names is a project on its
own, so picking two would mean shipping neither well.

What the renderer already has constrains the choice. Phase 29 established
order-2 spherical harmonics, cosine-convolved irradiance, probe influence
weighting and blending, and a six-face scene capture callback. Phase 26
established a physically based forward path whose ambient term is a single
irradiance vector per pixel. Phase 28 established reverse-Z shadow sampling.
Nothing in the repository unwraps lightmap UVs, and no asset in `assets/`
carries a second UV set.

The development hardware is an Intel HD Graphics 530, which reports
`D3D12_RAYTRACING_TIER_NOT_SUPPORTED`. Any ray-tracing path is therefore
unverifiable here on the GPU, and a path that cannot be exercised must not be
able to influence a frame on this machine.

## Decision

**Ship baked irradiance volumes.** Indirect diffuse radiance is precomputed by
a CPU path tracer and stored in a regular grid of order-2 spherical-harmonic
probes, each carrying an octahedral map of mean and mean-square visibility
distance. A surface
reads the volume with normal-weighted trilinear interpolation gated by a
Chebyshev visibility test. The forward path consumes the result as the diffuse
ambient term it already had.

**Do not ship lightmaps in this phase.** A probe volume lights static and
dynamic geometry by the same mechanism, needs no second UV set and no unwrapper,
and has memory proportional to the volume rather than to the surface area. A
lightmap's advantage — indirect detail at texel rather than probe spacing —
costs a UV pipeline, a seam-aware dilation pass, and a per-asset bake identity
this phase has no budget for. It stays available as later work; the baker's
integrator is the same one a lightmap bake would use.

**Do not ship screen-space or voxel GI as the primary method.** Screen-space GI
cannot light what the camera does not see, and the renderer already has the
screen-space tool that makes sense for it: Phase 29's reflection trace. Voxel GI
is a dynamic method whose cost lands at runtime, which is the wrong trade for
the static architecture the test scenes are built from, and whose leakage is
governed by voxel resolution that the development hardware cannot afford.

**Keep ray tracing behind a reported capability, and make the fallback the
default rather than the exception.** The RHI gains `Feature::RayTracing` and
`Feature::RayQuery`, reported from `D3D12_FEATURE_D3D12_OPTIONS5`: tier 1.0
reports the first, tier 1.1 reports both. `cyber::renderer` gains a pure
selection policy that maps a requested technique plus reported capabilities onto
the technique that will actually run, per effect, with a rasterized or
software-reference answer whenever the hardware path is absent. The policy owns
no device objects and is unit tested against fabricated capability sets, which
is how the logic is verified on hardware that cannot run it.

**Make the software ray tracer the reference, not a stand-in.** The BVH,
acceleration-structure split, and ray queries in `raytrace.h` are modelled on
DXR's own vocabulary: bottom-level structures own triangles, a top-level
structure owns instances with transforms, and the two query entry points are a
closest-hit and an any-hit traversal. A later DXR implementation then replaces
the traversal rather than the architecture, and the software path remains the
oracle its results are compared against.

## Consequences

### Positive

- One GI method, complete: bake, storage, interpolation, leak suppression,
  renderer integration, and measured acceptance.
- Runs on every target, the development adapter included, because the bake and
  the lookup are both CPU work and the lookup is a handful of taps.
- Reuses Phase 29's spherical-harmonic basis and Phase 26's ambient term, so
  the integration surface is one pointer on `PbrLightingData`.
- The acceleration structures, path tracer, and denoiser are the same pieces a
  hardware path needs, so the ray-tracing work is not thrown away.
- The capability gate is testable without capable hardware, and its default
  answer is the path that always exists.

### Negative

- Indirect lighting is baked: moving a wall or a light invalidates the volume,
  and nothing in this phase re-bakes incrementally.
- Probe spacing bounds indirect detail. Contact-scale indirect shadowing is not
  represented and must come from screen-space ambient occlusion.
- Bake time grows with probe count times samples times bounces. A production
  volume is minutes of CPU work, not milliseconds.
- `Feature::RayTracing` is reported but no RHI acceleration-structure or
  dispatch object exists yet, so a caller can learn that the hardware is capable
  and still have only the software path to use. The selection policy states this
  explicitly rather than implying a GPU path is available.

### Neutral

- Volume storage is 2,164 bytes per probe: nine SH coefficients of three
  channels (27 floats), a 16 x 16 octahedral map of two visibility moments (512
  floats), a back-face fraction, and a classification. A 16 x 8 x 16 volume is
  2,048 probes, about 4.4 MB. Most of that is visibility, and the resolution
  was measured rather than chosen: coarser maps leaked light through walls in
  the acceptance suite (see the design note).
- Job-system use changes bake scheduling only. The integrator's sample sequence
  is index-derived, so a bake is byte-identical with and without workers.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Baked lightmaps first | No UV unwrapper, no second UV set on any asset, and no seam or dilation tooling; dynamic geometry would need a probe volume anyway. |
| Screen-space GI first | Cannot light off-screen geometry, and the obvious screen-space trace already exists as Phase 29's reflections. |
| Voxel cone-traced GI first | Runtime cost and leakage both scale against the development hardware; the target scenes are static, which is the case it is worst at. |
| Irradiance probes without visibility moments | Light leaks through walls at any affordable probe spacing; the Chebyshev test is what makes a grid usable indoors. |
| Full DXR implementation now | Unverifiable on the development adapter; an unexercised GPU path in the frame is worse than a documented absence. |
| A single `Feature::RayTracing` flag | Inline ray queries and a ray-generation pipeline are separately available in D3D12 and a caller plans differently for each. |
| Capability gate inside the software renderer | Capabilities belong to the device; a software path that never consults an adapter would have to invent one. |

## Review

Accepted with the Phase 31 implementation. The decision covers the GI method,
the acceleration-structure vocabulary, and the capability gate. It does not
claim an implemented GPU ray-tracing path, and the second-machine verification
the build guide asks for remains outstanding for want of capable hardware.
