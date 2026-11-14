# 0027 — Environment-map ownership and borrowed lighting views

- **Status:** accepted
- **Date:** 2026-11-11
- **Deciders:** @TheCyberKiddd
- **Related:** Phase 29, [design note](../design/phase-29-ibl-reflections.md)

## Context

The software PBR path consumes image-based lighting through a lightweight
`EnvironmentLighting` struct containing non-owning pointers to independently
allocated resources. Loading an equirectangular image and preparing its source
cubemap, specular prefilter, diffuse irradiance, and BRDF lookup previously
required callers to coordinate multiple allocations and failure paths. A
partial setup could accidentally be published to rendering code.

## Decision

Keep the low-level cubemap, filter, and evaluation APIs public, and add a
move-only `EnvironmentMap` aggregate that owns the four resources. Build the
aggregate in a local candidate and return it only after every allocation,
projection, and filter succeeds. `Lighting()` creates a non-owning view;
document that the owner must outlive the view's use. `Load()` composes the
existing image loader with the same build path instead of duplicating image
decoding.

Environment resources remain allocator-owned CPU reference assets in this
phase. Scene capture is performed through the software renderer's six-face
capture callback and produces a cubemap consumed by the same owning build path.
GPU upload, cooked-cache identity, and asynchronous reload are not implied by
this API.

## Consequences

### Positive

- Offers one failure-aware path from a decoded/file image to complete PBR
  environment lighting.
- Prevents publishing a half-built environment and gives its resources one
  explicit lifetime owner.
- Preserves lower-level APIs for applications that share, cache, or cook
  individual resources.
- Keeps the software reference path independent of GPU adapter availability.

### Negative

- The aggregate owns duplicate source and filtered data; sharing requires
  callers to use the lower-level borrowed-resource API.
- `EnvironmentLighting` remains non-owning, so retaining it past the
  `EnvironmentMap` lifetime is invalid.
- Runtime CPU precomputation has significant time and memory cost at production
  resolutions.

### Neutral

- Build settings are explicit and bounded by existing cubemap, prefilter, and
  BRDF-LUT limits.
- Job-system use changes scheduling only; output remains deterministic.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Replace `EnvironmentLighting` with an owning descriptor | Breaks low-level composition and introduces ownership into the per-material/pixel-shader input. |
| Add a global environment cache/registry | Requires cache identity, invalidation, and synchronization policy not established by this phase. |
| Expose only independent resources | Leaves every caller to repeat error handling and risks partial publication. |
| Add GPU resource ownership now | No production GPU scene-rendering path consumes these resources yet. |

## Review

Accepted with the Phase 29 software-reference implementation. The API does not
establish GPU-resource or cooked-cache ownership.
