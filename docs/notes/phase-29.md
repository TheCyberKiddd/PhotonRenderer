# Phase 29 reading notes — image-based lighting and reflections

- IBL treats the surrounding environment as directional radiance, not a scalar
  ambient color.
- Lambertian diffuse can be pre-integrated into irradiance; its units must stay
  consistent with the renderer's linear radiance convention.
- GGX specular prefiltering stores roughness-dependent results in cubemap mips.
- The split-sum approximation separates directional environment radiance from
  the view/roughness BRDF response stored in a 2D lookup.
- Deterministic Hammersley sampling makes CPU precomputation reproducible and
  suitable for exact single-job/multi-job comparisons.
- Local probes require both a spatial influence weight and a parallax proxy;
  overlapping weights must be normalized or fall back without energy gaps.
- Screen-space reflections are incomplete by construction and must blend by
  confidence into probes rather than replace them unconditionally.
- The software renderer's reverse-Z depth convention must be preserved in the
  hierarchical depth representation and SSR intersection tests.
- Sources: [Filament PBR](https://google.github.io/filament/Filament.html),
  [LearnOpenGL diffuse IBL](https://learnopengl.com/PBR/IBL/Diffuse-irradiance),
  and [LearnOpenGL specular IBL](https://learnopengl.com/PBR/IBL/Specular-IBL).
