# Phase 26 design — physically based forward renderer

Date: 2026-11-08
Status: complete
Related: [Phase 26 build guide](../graphics/CYBERGRAPHICS_BUILD_GUIDE.md#phase-26-physically-based-forward-renderer), [Phase 26 summary](../graphics/PHASE_26.md), [reference renderer contract](../testing-reference-renderer.md)

## Scope and implementation boundary

Phase 26 adds a linear-HDR metallic-roughness BRDF and CPU forward pixel
shader to `cyber_software_renderer`. This is a separate model from the
existing Lambert/Blinn-Phong shader. Material values follow glTF's
metallic-roughness convention: base color is linear RGB at evaluation time,
metallic is in `[0, 1]`, roughness is perceptual roughness in `[0, 1]`, and
normal-map values are linear data. The renderer clamps roughness to `0.045` to
bound the GGX highlight.

The public CPU API is in [`pbr.h`](../../engines/cyber_software_renderer/include/cyber/software/pbr.h).
Directional, point, and spot lights are supported. Light colors are linear;
directional `radiance` is W/(m^2 sr), while point and spot `intensity` is
RGB radiant intensity in W/sr. Positions, ranges, and distances use meters.
Point and spot attenuation use inverse-square falloff multiplied by a smooth
finite-range fade. Spot angles are radians from the direction the light
travels. The ambient term is diffuse irradiance in W/m^2; specular environment
lighting and image-based lighting remain Phase 29 work.

## BRDF

For unit normal `N`, view direction `V`, and light direction `L`, define
`H = normalize(V + L)`, and `NoX = max(dot(N, X), 0)`. The forward response is
the Cook-Torrance sum:

```text
f_r = (1 - F) * (1 - metallic) * baseColor / pi
    + D_GGX * G_Smith * F / max(4 * NoV * NoL, epsilon)
L_out = f_r * L_in * NoL
```

The implementation uses isotropic Trowbridge-Reitz GGX with
`alpha = roughness^2`:

```text
D_GGX = alpha^2 /
        (pi * (NoH^2 * (alpha^2 - 1) + 1)^2)
```

Smith visibility is the product of two exact separable Smith-GGX terms:

```text
G1(NoX) = 2 * NoX /
          (NoX + sqrt(alpha^2 + (1 - alpha^2) * NoX^2))
G_Smith = G1(NoV) * G1(NoL)
```

Schlick Fresnel is `F = F0 + (1 - F0) * (1 - VoH)^5`, where
`F0 = lerp(0.04, baseColor, metallic)`. Diffuse energy is reduced by the
Fresnel response and metallic factor. Inputs are normalized at the API boundary;
back-facing or non-finite BRDF parameters contribute zero.

Tangent-space normal maps use a re-orthogonalized interpolated tangent and the
interpolated bitangent sign. Tangents are transformed with the model's linear
transform, normals with its inverse-transpose matrix. The caller/importer is
responsible for supplying valid tangents and decoding authored sRGB base-color
images to linear before texture storage. Normal maps remain linear data.

## HDR output

Lighting remains unclamped linear HDR in the framebuffer. The Phase 26 sample
applies the fitted ACES curve with exposure measured in stops; the image writer
then encodes RGB to sRGB. Alpha is preserved and is not tone-mapped or
sRGB-encoded. The curve is an output transform, not part of BRDF evaluation.

## Validation and limits

`software_pbr` covers BRDF hemisphere behavior, a deterministic white-furnace
integration across four roughness values, tangent-frame handedness and
degeneracy, directional/point/spot response, inverse-square attenuation,
tone-map exposure, and the null-shader-data fallback.

The `04_lighting` sample rasterizes a tessellated, normal-mapped 7x7 sphere
grid and compares it with an analytic-sphere CPU path tracer. The reference
owns separate BRDF equations and deterministic cosine/GGX sampling, and
evaluates shadowed direct lights plus environment bounces. It shares only scene
inputs and material/light parameters with the forward path.

The sample defaults to 8x supersampling per axis and box-filters to its 480x360
output. On Windows 10 22H2, Intel Core i5-6400, and MSVC 19.44, the default
four-bounce, 64-path-sample/output capture measured `0.00879` mean sRGB
absolute error and `0.030%` coverage mismatch. The 64 path samples are
distributed over each pixel's 8x8 subpixel grid. The acceptance limits are
`0.015` mean error and `0.1%` coverage mismatch. See the [Phase 26 report](../reports/phase-26.md)
and [reference contract](../testing-reference-renderer.md) for reproduction
commands and test coverage.

## Sources

- Khronos, [glTF 2.0 material model](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materials).
- Brent Burley, [Physically Based Shading at Disney](https://disneyanimation.com/publications/physically-based-shading-at-disney/).
- Eric Heitz, [Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs](https://jcgt.org/published/0003/02/03/).
- Bruce Walter et al., [Microfacet Models for Refraction through Rough Surfaces](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf).
