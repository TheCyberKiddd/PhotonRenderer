# Phase 29 design — image-based lighting and reflections

Date: 2026-11-11
Status: Complete for the software-reference scope: IBL, local probes,
hierarchical SSR, procedural sky, six-face scene-driven probe capture, and
measurable acceptance are implemented.
Related: [Phase 29 roadmap](../graphics/CYBERGRAPHICS_ROADMAP.md#phase-29-image-based-lighting-and-reflections), [build guide](../graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-29-image-based-lighting-and-reflections), [ADR 0027](../adr/0027-environment-map-ownership.md)

## Scope and boundaries

Phase 29 provides the project's deterministic software reference path for
image-based lighting and reflections. GPU scene rendering and asset-cooker
integration are outside this phase's software-renderer contract. Callers can
load an equirectangular image, provide a cubemap, or capture a scene through
six software-rendered probe views. SSR consumes the current frame's HDR color
and reverse-Z depth.

## Environment representation and ownership

`Cubemap` owns linear RGBA32F radiance with a bounded mip chain. The public
`EnvironmentMap` owns source, GGX-prefiltered specular, cosine-convolved diffuse
irradiance, and split-sum BRDF resources. `EnvironmentMap::Create` accepts a decoded equirectangular framebuffer or
linear-radiance cubemap; `EnvironmentMap::Load` uses the image loader,
including Radiance RGBE HDR. Creation is transactional: no lighting view is
returned unless projection and each precomputation succeeds.

`EnvironmentMap::Lighting` is a borrowed view, not a copy. The owning map must
outlive every pixel-shader use of that view. Lower-level APIs remain available
for applications with their own cache, cooking, or shared-resource lifetime.

`CaptureReflectionProbe` invokes a caller-owned scene-render callback for all
six cubemap directions. Each invocation receives a 90-degree reverse-Z camera,
a cleared RGBA32F target, and a cleared depth target. The returned cubemap is
published only after every face succeeds. `EnvironmentMap::Create` can consume
that cubemap, regenerates its source mip chain, and performs the usual
specular, diffuse, and BRDF precomputation. Callers own scene traversal,
materials, direct lighting, capture scheduling, and refresh policy.

Radiometric values are linear RGB. Source maps store radiance, diffuse maps
store cosine-convolved irradiance, and evaluation returns outgoing radiance.
The diffuse path uses solid-angle-weighted order-2 spherical harmonics or an
irradiance cubemap. Specular uses deterministic GGX importance sampling,
solid-angle-matched source mips, and a BRDF LUT with a deterministic analytic
fit when the table is omitted. White-furnace tests guard energy conservation.

## Local and screen-space reflections

Reflection probes borrow an `EnvironmentLighting` set, an influence AABB, and
an optional parallax proxy AABB. Smoothstep edge weights are normalized when
probes overlap; any uncovered weight is supplied by the optional fallback.
Box parallax correction redirects the specular sample to the proxy
intersection. PBR selects one probe set or one global environment instead of
summing two complete environment responses.

SSR reads the current linear HDR framebuffer and reverse-Z depth buffer. An
optional closest-depth Hi-Z pyramid skips empty regions; the same trace can
run without it. Hit confidence accounts for screen-edge/grazing/roughness
limits, and the PBR path blends the result over the probe response by
confidence. Misses therefore retain the probe lighting.

## Procedural sky

The CPU sky evaluates single-scattering Rayleigh and Mie atmosphere radiance,
ground response, and an optional solar disk, then can capture the result into a
cubemap for the same environment precomputation path. This is a deterministic single-scattering reference implementation.

## Validation strategy and limits

Unit tests cover cube addressing/seams, solid angles, deterministic
precomputation, constant-environment invariants, the BRDF table, six-face
scene capture, depth ordering and capture failure, probe weights and parallax,
Hi-Z reduction, SSR descriptors, sky behavior, and RGBE file decoding.
Acceptance tests compare a rendered metallic sphere with analytic reflections,
measure probe transitions along a camera path against a no-blending control,
find a reflected panel with SSR, and render a white furnace over
roughness/metalness combinations.

Public API ownership is recorded in [ADR 0027](../adr/0027-environment-map-ownership.md).
The completed scope is validated against the software reference renderer and
makes no claim of GPU-rendered Unreal Engine parity.

## Sources

- [Filament: Physically Based Rendering](https://google.github.io/filament/Filament.html)
- [LearnOpenGL: Diffuse irradiance](https://learnopengl.com/PBR/IBL/Diffuse-irradiance)
- [LearnOpenGL: Specular IBL](https://learnopengl.com/PBR/IBL/Specular-IBL)
- [Unreal Engine: Reflections Environment](https://dev.epicgames.com/documentation/en-us/unreal-engine/reflections-environment-in-unreal-engine)
