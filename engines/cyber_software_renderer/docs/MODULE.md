# cyber_software_renderer — module contract

**Kind:** product
**Layer:** 4
**Order:** 41
**Status:** active
**Owner:** @TheCyberKiddd

## Purpose

CPU reference renderer: framebuffer, rasterization, depth, textures, shaders,
clustered PBR reference shading, golden images, and the second plugin. In Phase
8 it owns the linear framebuffer,
sRGB present conversion, image export, frame pacing, the bitmap-font overlay,
line drawing, the tile-parallel edge-function triangle rasterizer, the
reverse-Z depth buffer, the clipped and culled vertex pipeline, the orbit and
fly cameras, and perspective-correct interpolation. Phase 9 adds textures and
CPU shaders.

## Public API

`include/cyber/software/`

| Header | Contents |
|---|---|
| `framebuffer.h` | `PixelFormat`, `FramebufferDesc`, `Framebuffer`, `PixelSizeBytes`, `QuantizeUnorm8`, `EncodeSrgb8`. |
| `present_target.h` | `PresentByteSize`, `PackPresentPixels`. |
| `image_io.h` | `ImageFormat`, `ImageWriteDesc`, `EncodeImage`, `WriteImage`. |
| `debug_text.h` | 5x7 glyph metrics, `TextMetrics`, `MeasureText`, `DrawText`. |
| `frame_pacer.h` | `FramePacer`. |
| `raster.h` | `RasterVertex`, `RasterTriangle`, `RasterizerDesc`, `RasterStats`, `DrawLine`, `FillTriangle`, `RasterizeTriangles`, `DepthRasterVertex`, `DepthRasterTriangle`, `RasterizeDepthTestedTriangles`, `ShadedRasterVertex`, `ShadedRasterTriangle`, `RasterizeShadedTriangles`, `DepthOnlyRasterVertex`, `DepthOnlyRasterTriangle`, `RasterizeDepthOnlyTriangles`. |
| `depth_buffer.h` | `DepthBufferDesc`, `DepthBuffer`. |
| `camera.h` | `Camera`, `CameraForward`, `CameraRight`, `CameraUp`, `CameraView`, `CameraProjection`, `FlyCamera`, `FlyCameraInput`, `UpdateFlyCamera`, `OrbitCamera`, `OrbitCameraInput`, `UpdateOrbitCamera`, `OrbitCameraPosition`, `OrbitCameraView`, `OrbitCameraProjection`. |
| `pipeline.h` | `Vertex`, `Viewport`, `CullMode`, `FrontFace`, `PipelineDesc`, `PipelineStats`, `ClipVertex`, `ClipPolygon`, `ClipTriangleNearPlane`, `DrawTriangles`. |
| `shading.h` | `VertexInput`, `Varyings`, `VertexStageOutput`, `VertexShaderFn`, `PixelShaderFn`, `LerpVaryings`. |
| `shader.h` | `ShaderDesc`, `TransformShaderData`, `TransformVertexShader`, `DrawShadedTriangles`, `DepthOnlyVertexShaderFn`, `DepthOnlyShaderDesc`, `TransformDepthOnlyVertexShader`, `DrawDepthOnlyTriangles`. |
| `texture.h` | `TextureFormat`, `AddressMode`, `FilterMode`, `SamplerDesc`, `TextureDesc`, `Texture`, `SampleTexture`, `SampleTextureLevel`. |
| `pbr.h` | `PbrMaterial`, point/directional/spot lights, `PbrLightingData`, `EvaluateGgxBrdf`, `PbrPixelShader`, `ToneMapAces`. |
| `shadow.h` | Reverse-Z `ShadowMapView`, directional/spot/point projection helpers, practical cascade fitting and overlap sampling, CPU shadow atlas allocation, filtering and debug helpers, `RenderShadowMap`, `RenderDirectionalShadowMaps`, `EvaluateShadowVisibility`. |
| `shadow_quality.h` | `ShadowProbe`, `ShadowAcneMetrics`, `MeasureShadowAcne`, `ShadowContactMetrics`, `MeasureShadowContact`, `ShadowTransectMetrics`, `MeasureShadowTransect`, `ShadowStabilityMetrics`, `MeasureShadowStability`. |
| `clustered_pbr.h` | `ClusteredPbrLightingData`, `BuildClusterLightBounds`, `ClusteredPbrPixelShader`. |
| `antialias.h` | `AntiAliasQuality`, `ResolveFilter`, `SupersampleFactorForQuality`, `SampleCountForQuality`, `ResolveFilterRadius`, `EvaluateResolveFilter`, `ResolveDesc`, `ResolveSupersampled`, `AntiAliasTargetDesc`, `AntiAliasTarget`. |
| `ibl.h` | `CubeFace`, `CubemapDesc`, `Cubemap`, `CubeDirectionFromFaceUv`, `CubeTexelSolidAngle`, `SampleCubemapNearest`, `SampleCubemapFace`, `SampleCubemapLevel`, `EquirectangularProjectionDesc`, `ProjectEquirectangularToCubemap`, `SphericalHarmonicsL2`, `ProjectCubemapToSphericalHarmonics`, `EvaluateSphericalHarmonicsIrradiance`, `IrradianceFilterDesc`, `BuildDiffuseIrradianceCubemap`, `SpecularPrefilterDesc`, `SpecularMipRoughness`, `SpecularRoughnessMip`, `PrefilterSpecularCubemap`, `BrdfLutDesc`, `BuildSplitSumBrdfLut`, `SampleSplitSumBrdf`, `IblSurface`, `EnvironmentLighting`, `EnvironmentMapDesc`, `EnvironmentMap`, `ComputeSpecularOcclusion`, `EvaluateIblDiffuse`, `EvaluateIblSpecular`, `EvaluateIblSpecularDirection`, `EvaluateImageBasedLighting`, `ApplySpecularEnvironmentWeight`, `ReflectionProbe`, `ReflectionProbeSet`, `ReflectionProbeCaptureDesc`, `ReflectionProbeCaptureView`, `ReflectionProbeCaptureFn`, `CaptureReflectionProbe`, `EvaluateReflectionProbeWeight`, `ApplyBoxParallaxCorrection`, `EvaluateReflectionProbes`, `HiZPyramid`, `ScreenSpaceReflectionDesc`, `ScreenSpaceReflectionSample`, `TraceScreenSpaceReflection`, `ResolveReflections`, `ProceduralSkyConfig`, `EvaluateProceduralSky`, `EvaluateSunTransmittance`, `RenderProceduralSkyCubemap`. |
| `post_process.h` | `Luminance`, `LinearizeReverseZ`, `PostProcessView`, `ViewGeometry`, `MakeViewGeometry`, `ReconstructViewPosition`, `ExposureMode`, `LuminanceHistogram`, `ExposureSettings`, `ExposureState`, `BuildLuminanceHistogram`, `HistogramAverageLuminance`, `ExposureEvForAverageLuminance`, `AdaptExposure`, `ComputeExposureEv`, `ToneMapOperator`, `ToneMapSettings`, `ApplyToneMap`, `ColorGradeSettings`, `ApplyColorGrade`, `ColorGradeLut`, `BloomSettings`, `AmbientOcclusionSettings`, `DepthOfFieldSettings`, `MotionBlurSettings`, `PostProcessPass`, `PostProcessPassName`, `PostProcessSettings`, `PassTiming`, `PostProcessTimings`, `PostProcessInputs`, `PostProcessStack`. |
| `post_process_quality.h` | `ImageDifferenceMetrics`, `MeasureImageDifference`, `GhostingMetrics`, `MeasureGhosting`, `TemporalStabilityMetrics`, `MeasureTemporalStability`, `EdgeAliasingMetrics`, `MeasureEdgeAliasing`. |
| `temporal_aa.h` | `AntiAliasMethod`, `AntiAliasMethodName`, `RadicalInverse`, `HaltonPoint`, `TemporalJitterOffset`, `ApplyJitterToProjection`, `TemporalAaSettings`, `TemporalAaInputs`, `TemporalAaStats`, `TemporalAaResolver`, `FxaaSettings`, `FxaaResolve`. |
| `lighting.h` | `DirectionalLight`, `PointLight`, `SurfaceMaterial`, `LightingData`, `LambertBlinnPhongPixelShader`. |
| `mesh.h` | `MeshVertex`, `Mesh`, `ObjMesh`, `LoadObjFromMemory`, `LoadObjFile`. |
| `golden_image.h` | `GoldenImageStatus`, `GoldenImageDesc`, `GoldenImageResult`, `CompareImages`, `WriteDifferenceImage`. |
| `software.h` | Umbrella include. |

Only headers under this path are public. Everything under `src/` is private.
Public headers are self-contained, never include OS headers, and never include
another module's private headers.

### Failure and ownership

- All functions are non-throwing. Recoverable failures return
  `Result<T, Error>` or `Status<Error>`; invalid arguments are reported as
  `ErrorCode::InvalidArgument`.
- `Framebuffer` is move-only and owns allocator-backed storage. `Create` is the
  only operation that allocates. Destruction releases the block through the
  allocator passed to `Create`.
- Pixel get/set outside the image is safe: reads return transparent black and
  writes are ignored. Rectangle operations clip.
- `EncodeImage` returns an allocator-backed `DynamicArray<Byte>`; `WriteImage`
  writes through `cyber_platform` and owns nothing.
- `DrawText`, `DrawLine`, `FillTriangle`, and `FramePacer` are allocation-free.
- `RasterizeTriangles` and `RasterizeDepthTestedTriangles` allocate temporary
  prepared-triangle and tile-bin arrays from the caller's allocator and release
  them before returning. Workers do not touch the allocator. Invalid targets and
  null spans return `ErrorCode::InvalidArgument`; a triangle or bin count beyond
  32 bits returns `ErrorCode::Overflow`; allocation failure returns
  `ErrorCode::Internal`.
- `DepthBuffer` is move-only and owns allocator-backed storage. `Create` is the
  only operation that allocates. Depth is a float reverse-Z value: 1 at the near
  plane, 0 at the far plane, cleared to 0, and tested greater-or-equal.
- `DrawTriangles` allocates nothing; it clips into a fixed-capacity local
  polygon and hands the result to the depth-tested rasterizer with the caller's
  allocator. It validates the color target, the depth target dimensions, the
  viewport, and the vertex count, and returns `ErrorCode::InvalidArgument` for
  a mismatch.
- `ClipTriangleNearPlane` is allocation-free and returns a value.
- The camera values and controllers are allocation-free plain data.
- The Phase 27 clustered PBR shader consumes backend-neutral CPU cluster lists.
  Its CPU differential test compares against all-light shading; it is not a GPU
  implementation or a GPU performance claim.
- Phase 28's software shadow path renders maps through a depth-only clipped
  reverse-Z raster path that takes no color target and runs no pixel shader,
  and samples them with constant, slope-scaled, and normal-
  offset bias plus bounded PCF and contact-hardening PCSS. Optional stratified
  16-tap filtering, point-face seam blending, and default-off height fog with
  shadowed light scattering are also available. Directional PBR lights support
  practical-split cascades with texel-snapped fitting and overlap blending.
  `RenderDirectionalShadowMaps` validates a complete set of same-sized targets,
  renders every cascade, and only then publishes the borrowed depth-map
  pointers to the lighting set; a failed render never exposes a partially
  updated set. Spot and point lights use caller-owned maps. Helpers provide spot/point
  projection transforms, a guarded fixed-tile CPU atlas, importance-based
  resolution selection, and cascade visualization. The RHI module provides a
  depth-only render-graph helper, reverse-Z comparison-sampler descriptor
  helper, and shadow-depth resource lifecycle; D3D12 rasterization/readback and
  reverse-Z comparison sampling are tested. `shadow_quality.h` measures the
  four shadow artifacts — acne, detached contact shadows, cascade-split steps,
  and camera-motion shimmer — through the shipped sampler, and the Phase 28
  acceptance test turns those measurements into thresholds on a rendered
  four-cascade scene. Each threshold is paired with a deliberately mis-biased
  or unblended control so a pass proves the metric can fail. The metrics are
  CPU analysis tools: they define no scene and no limit, and they do not cover
  the GPU path. GPU scene lighting/scheduling remains outstanding.
- Phase 28's `AntiAliasTarget` is move-only and owns a supersampled colour and
  depth pair sized to the resolved dimensions times its factor; `Create` is the
  only operation that allocates. A caller renders through the normal pipeline
  at `RenderViewport()` and resolves into a destination that must be exactly
  `Width() x Height()`. Supersampling does not change the aspect ratio, so
  camera setup written against the final dimensions stays correct. The factor
  is bounded to [1, 16] and the supersampled dimensions to 16384.
  `ResolveSupersampled` allocates a temporary band buffer from the caller's
  allocator and releases it before returning; its output is byte-identical
  with and without a job system and for any band size. The renderer-wide
  default is `AntiAliasQuality::High`, a 4x4 factor with Mitchell-Netravali
  reconstruction, filtered in linear light.
- Phase 29's `Cubemap` and `EnvironmentMap` are move-only and own
  allocator-backed storage, and `Create` is the only operation that allocates.
  Out-of-range faces, levels, or texel coordinates read transparent black and
  ignore writes, as framebuffer access does. `EnvironmentMap` construction is
  transactional: the source radiance, prefiltered specular, diffuse irradiance,
  and split-sum BRDF resources are published together or not at all, so a
  failed precomputation never yields a partially built environment.
  `EnvironmentMap::Lighting` returns a **borrowed** `EnvironmentLighting` view
  of those resources, not a copy; the owning map must outlive every
  pixel-shader use of it. The same holds for the probe pointers inside a
  `ReflectionProbeSet`. `CaptureReflectionProbe` invokes a caller-owned render
  callback for all six faces and returns a cubemap only after every face
  succeeds; the caller owns scene traversal, materials, direct lighting, and
  refresh policy. Radiometric values are linear RGB: source maps store
  radiance, diffuse maps store cosine-convolved irradiance, and evaluation
  returns outgoing radiance. See [ADR 0027](../../../docs/adr/0027-environment-map-ownership.md).
- Phase 30's `PostProcessStack` is move-only and owns the ping-pong HDR pair,
  the occlusion buffer, and the bloom chain for one resolution. `Create` is the
  only operation that allocates: no pass allocates while a frame is in flight.
  `Execute` validates that the destination matches the stack's size and
  returns `ErrorCode::InvalidArgument` otherwise; a pass that is disabled or
  whose input buffer is absent is skipped and reported as not executed, which
  is not an error. Exposure state lives in the stack, because adaptation is a
  property of the camera over time rather than of one frame. `ColorGradeLut` is
  move-only and owns its entries. Reported pass times are CPU wall time from
  the platform monotonic clock, not GPU timestamps.
- `TemporalAaResolver` is move-only and owns the color history, the two
  reprojected drift buffers, and the depth history; `Create` is the only
  operation that allocates. The first frame after `Create` or `Reset` is copied
  through unchanged. A supplied depth buffer enables disocclusion rejection
  from the following frame, once there is a previous depth to compare against.
  Both the resolve and the whole post chain write byte-identical output with
  and without a job system.
- `post_process_quality.h` reads borrowed images, allocates nothing, and is
  deterministic. Like `shadow_quality.h` it defines no scene and no limit; the
  thresholds live in the acceptance test, and each is paired with a control
  configuration that fails it.
- Phase 31's acceleration structures are move-only and allocator-owned, and
  `Build` is the only operation that allocates; queries are `const`, write
  nothing but the caller's output, and may run concurrently. A
  `TopLevelAccelerationStructure` **borrows** its bottom-level structures,
  which must outlive it and must not move. Builds are deterministic and keep
  degenerate triangles, so primitive indices always match the input. A query
  never recurses: the builder bounds depth at `kMaxBvhDepth`. See
  [ADR 0028](../../../docs/adr/0028-gi-approach.md).
- `IrradianceVolume` is move-only and owns its probes. `BakeIrradianceVolume`
  is transactional: probes are baked into scratch storage and copied in only
  when every probe has finished, so a rejected bake leaves the volume as it
  was. Probes classified `Inside` and probes still `Unbaked` never contribute
  to a lookup. `GiScene` and `IrradianceVolumeView` are borrowed views; the
  geometry, materials, lights, and volume must outlive every call that reads
  them. Radiometric units match the rest of the renderer, so a volume baked
  from a set of lights and sampled by `PbrPixelShader` reproduces the indirect
  light those lights produce.
- `SpatioTemporalDenoiser` is move-only and owns its history and scratch for
  one resolution; `Create` is the only operation that allocates. History is
  updated only by a successful `Denoise`, and `Reset` discards it. Guide
  conventions are the post chain's.
- `gi_quality.h` reads borrowed data, allocates nothing, and is deterministic,
  with thresholds living in the acceptance suite beside the controls that
  fail them.

## Dependencies

Allowed: cyber_core, cyber_math, cyber_memory, cyber_platform, cyber_jobs,
cyber_renderer

Forbidden: cyber_rhi, cyber_graphics

Dependencies are one-way. A module may depend only on modules with a lower
`order` value, or on a declared contract such as `cybergraphics_api`.

`cyber_assets`, `cyber_scene`, and `cybergraphics_api` join this list when
Phases 9 and 25 need them. They have no targets yet and are not declared early.

## Invariants

- No direct RHI dependency or graphics-backend headers; rendering is CPU-only.
- `cyber_renderer` is used only for the backend-neutral clustered light-list
  contract; software shading and rasterization remain CPU-only.
- Color is stored linear. sRGB encoding happens only in `PackPresentPixels` and
  `EncodeImage`, never in framebuffer storage.
- Deterministic output for golden images: no timestamps, no random seeds, no
  locale-dependent formatting, and no address-dependent behavior.
- Rasterization snaps vertices to 24.8 fixed point, uses integer edge functions
  with the top-left rule, and bins triangles into screen tiles. Worker count and
  tile size never change the image bytes.
- Clip space is `-w..w` in x and y and `0..w` in z. Only the near plane `z = w`
  is clipped; the far plane is rejected by the reverse-Z test, and the side
  planes use a `+/-16384`-pixel guard band that drops and counts the triangle.
- Depth is perspective-correct NDC depth: interpolated barycentrically, tested
  greater-or-equal, and written only when the test passes. Colors use
  `attribute / w` and `1 / w`; the Phase 7 affine functions keep their original
  arithmetic and output.
- The viewport maps NDC to screen with a y flip; positive screen-space signed
  area is clockwise and negative is counter-clockwise.
  `FrontFace::CounterClockwise` is the default front face for right-handed
  geometry, matching D3D12's `FrontCounterClockwise = TRUE`. Culling happens
  after clipping and before binning.
- Every runtime buffer uses an explicit allocator.
- Row access is stride-aware; padding bytes are never written by pixel
  operations.
- Intended as the CPU implementation of the stable API in
  `include/cybergraphics/`; the plugin adapter is Phase 25 work.

## Non-goals

- Performance parity with the GPU.
- Production rendering, HDR, tone mapping, and color management beyond sRGB.
- General image transcoding and PNG decoding; Phase 9 reads PPM and TGA and writes them, and Phase 24 adds cooked assets.
- vsync, swapchains, and GPU presentation (`cyber_rhi`, Phase 13).
- The stable plugin ABI itself; it is defined in `include/cybergraphics/`.

## Tests

`software_framebuffer`: descriptor validation, formats, stride padding, clear,
pixel access, clipping, fill, blit, conversion, moves, and allocator balance.
`software_present`: sRGB packing, alpha, explicit stride, determinism, and
validation.
`software_image_io`: exact PPM/TGA headers, pixel order, sRGB versus linear
encoding, deterministic in-memory encoding, file round-trip, and error paths.
`software_text`: metrics, glyph coverage, scaling, clipping, and blank glyphs.
`software_frame_pacer`: deadline planning, fixed-grid advance, stall resync, and
statistics.
`software_raster`: all-octant, reversed, degenerate, non-finite, and clipped
lines; the top-left fill rule on shared edges and vertices; winding and color
interpolation; invalid geometry and guard-band rejection; statistics; and
RGBA8 quantization.
`software_raster_coverage`: a 10,000-triangle jittered mesh covers 640x360
exactly once; single-threaded and job-system outputs are byte-identical across
tile sizes 16/64/128 and one/four workers.
`software_screenshot`: fixed scene encoded twice is byte-identical; PPM and TGA
files read back equal.
`software_depth_buffer`: descriptor validation, stride padding, reverse-Z
clear, greater-or-equal test/write, out-of-range safety, and moves.
`software_camera`: camera basis and view/projection mapping, fly movement,
frame-rate independence, pitch and distance clamps, and orbit zoom.
`software_texture`: descriptor validation, mip dimensions, wrap modes, nearest and
bilinear sampling, trilinear blending, both storage formats, and allocator
balance.
`software_shader`: vertex-stage output, perspective-correct varying interpolation,
clipped UVs and normals, and a constant and a pattern pixel shader.
`software_lighting`: Lambert falloff, Blinn-Phong specular, point attenuation,
ambient, and texture modulation.
`software_image_load`: PPM P6/P5/P3, comments, malformed headers, TGA 24/32-bit
origins, and unsupported rejection.
`software_mesh`: OBJ positions, normals, UVs, fan triangulation, negative
indices, and malformed faces.
`software_golden_image`: matching images, tolerance enforcement, differing-pixel
reporting, and difference-image output.
`software_golden_scenes`: five fixed 640x360 scenes, deterministic re-render, and
comparison against committed references.
`software_pipeline`: viewport transform and y flip, near-plane clipping output,
behind-camera rejection, winding and culling, depth ordering, perspective-correct
color, clipped-cube determinism, and invalid-input handling.
`software_depth_cube`: the fixed cube renders with hidden faces rejected, and
serial, parallel, and tile-size variants produce identical color and depth
bytes.
`software_public_headers`: every public header compiles together.

The visible checks are `cyber_sample_raster` and `cyber_sample_depth`. Both
`--capture` modes write and compare two screenshots of a fixed frame. The Phase
6 `cyber_sample_gradient` remains the framebuffer/present sample.

## Budgets

- Present packing and image export are O(width * height) with one pass and no
  per-pixel virtual calls.
- Rasterization is O(pixels covered + triangle-tile references) with integer
  edge stepping in the inner loop. The Phase 7 path has no per-pixel division;
  the depth-tested path has exactly one for perspective-correct color.
- The vertex pipeline transforms one matrix-vector per vertex and clips at most
  one triangle into two before binning. Per-pixel work is coverage, one depth
  compare, and the interpolation divide.
- Steady-state frame loops do not allocate: the sample reuses one present buffer
  and resizes it only on a size change. `RasterizeTriangles` allocations are
  caller-owned scratch and bounded by triangle and tile counts.
- Framebuffer memory is `stride * height` bytes; `RGBA32F` costs 16 bytes per
  pixel.
- Golden-image suite runtime budget: small test resolutions, no large assets.

## Roadmap

Phases 6, 7, 8, 9, 25, 26, 27, 28, 29, 30.
