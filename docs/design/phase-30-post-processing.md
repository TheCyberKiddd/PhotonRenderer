# Phase 30 design — post-processing and anti-aliasing

Date: 2026-11-13
Status: Complete for the software-reference scope: the HDR post chain, the
three anti-aliasing paths, the image-quality metric API, and the measured
acceptance suite are implemented.
Related: [Phase 30 roadmap](../graphics/CYBERGRAPHICS_ROADMAP.md#phase-30-post-processing-and-anti-aliasing),
[Phase 30 report](../reports/phase-30.md),
[Phase 28 design](phase-28-shadows.md),
[Phase 29 design](phase-29-ibl-reflections.md)

## Scope and boundaries

Phase 30 adds the passes that run between shading and display, and the
anti-aliasing the renderer resolves with. It is the software reference path, as
Phases 28 and 29 were: the passes shade on the CPU, the timings reported are
CPU wall time for the whole effect rather than a GPU queue timestamp, and GPU
implementations of these passes are not part of this phase.

In scope: automatic and manual exposure with eye adaptation, four tone-mapping
operators, lift/gamma/gain colour grading with a baked 3D lookup table, bloom,
screen-space ambient occlusion, depth of field, motion blur, per-pass runtime
toggles with per-pass timing, FXAA, temporal anti-aliasing with camera jitter
and velocity reprojection, and a measurement API with an acceptance suite over
it.

Out of scope: GPU versions of any of these passes, GTAO, temporal upsampling,
and automatic method selection. MSAA is not implemented and is not planned for
this renderer: a software rasterizer has no fixed-function coverage mask to
resolve, so the mechanism MSAA names does not exist here. The renderer's
supersampled path in `antialias.h` is what stands in its place, and
`AntiAliasMethod::Supersample` is how a caller selects it.

## The chain

`PostProcessStack` owns the scratch images for one resolution and runs the
enabled passes in a fixed order:

| Pass | Reads | Why it is where it is |
|---|---|---|
| Ambient occlusion | depth, normals | Acts on scene-referred radiance, before any lens effect |
| Depth of field | depth | A lens effect on the radiance that reached the sensor |
| Motion blur | velocity | The same, integrated over the shutter |
| Bloom | colour | Sensor and lens scatter, still scene-referred |
| Exposure | colour | The camera response begins here |
| Tone map | colour | Scene-referred radiance becomes display-referred |
| Colour grade | colour | A creative pass on the display-referred result |

The order is physical rather than configurable. Grading before tone mapping
would grade in an unbounded domain where a lift has no defined meaning;
blooming after tone mapping would scatter clipped values instead of the energy
that produced them.

Every pass is skipped outright when it is switched off or when the buffer it
needs is absent: it reads nothing, writes nothing, and reports zero time, while
still appearing in the timing table so a readout does not reflow as effects are
toggled. A missing input is not an error. Tone mapping and grading share one
resolve pass, because applying a baked lookup table costs nothing extra once
the pixel is already in registers; the measured cost of that pass is split
evenly between whichever of the two ran, so the parts still sum to the whole.

Determinism is the same contract as the rest of the renderer: sample patterns
come from fixed low-discrepancy sequences rather than a random source, and the
job-parallel and single-threaded paths write identical bytes.

## Exposure

Metering builds a 64-bin log2-luminance histogram and averages the samples
between two percentiles, discarding the darkest half and the brightest
twentieth. The percentile window is what keeps a small bright window or a black
doorway from dragging the whole frame. Pixels outside the histogram range land
in the end bins rather than being skipped, so the percentile walk is
total-preserving. Adaptation ramps toward the metered value at separate rates
up and down, snapping on the first frame after a reset.

## Anti-aliasing

Three paths, selected by `AntiAliasMethod`:

- **Supersample** — the renderer-wide N x N path. It resolves geometry,
  shading, shadow, and texture aliasing alike, at a cost that scales with the
  sample count.
- **FXAA** — a single-frame luminance edge filter. It touches only edges, so it
  cannot fix specular or texture aliasing. It is the cheap option and the
  fallback for a frame with no history.
- **Temporal** — jittered accumulation with reprojected, rectified history.

The temporal path jitters the projection by a Halton(2, 3) offset each frame
and accumulates into a history buffer reprojected along the velocity buffer.
History is constrained four ways: it is clipped into the colour range of the
current frame's 3x3 neighbourhood in YCoCg; the clip box is bounded by the
neighbourhood's variance as well as its extent; a pixel whose reprojection
lands off-screen or whose motion exceeds the disocclusion threshold drops its
history entirely; and, when a depth buffer is supplied, a pixel whose
reprojected depth does not match the surface rendered there drops it too.

### Depth is what makes a disocclusion visible

A velocity buffer cannot describe a disocclusion, and that is not a defect in
the buffer. A pixel of floor that a post was covering last frame has the
floor's velocity, which is correct; it reprojects to a pixel that the post
occupied, which is also correct; and the colour it finds there is the post's.
The result is a trail of the occluder smeared along the motion, and it is the
ghost that survives neighbourhood rectification most readily, because an
occluder edge is exactly where the neighbourhood is widest and the box is
therefore loosest.

Comparing the reprojected depth against the depth actually rendered closes it:
a surface that is not at the distance its history says it was is not the same
surface. The resolver keeps the previous frame's linearized view distance,
reprojects it with the same bilinear filter the colour history uses, and drops
history where the two disagree by more than `depthRejectRelative` of the
distance — a relative tolerance, because both depth precision and the parallax
one frame of motion produces scale with distance. The test starts one frame
after depth is first supplied, since it needs something to compare against,
and `TemporalAaStats::depthRejectedPixels` separates this from a camera simply
moving fast.

This was found by the sample rather than by the acceptance suite. The analytic
acceptance scene has no occlusion in it — it is a scrolling pattern, which is
what makes its ground truth and its velocity exact — so it cannot produce the
artifact and did not catch it. The 3D sample showed the slats trailing across
the floor on sight. The suite now covers the case as a unit test with an
occluder sliding across a background, in both configurations: without depth
the uncovered pixel keeps more than 30% of the occluder's brightness, and with
it, under 15%.

### Two decisions that the measurements, not the textbook, settled

**Rectification clamps per channel rather than clipping toward the box
centre.** Clipping along the segment from the box centre to the history sample
is the textbook form and preserves hue. It is the wrong choice here: the two
chroma axes of a near-grey neighbourhood are nearly degenerate, so the smallest
chroma disagreement dominates the clip ratio and drags luminance to the centre
of the box along with it. Measured on the acceptance scene, clipping cost about
a third of the converged accuracy the per-channel form keeps.

**Luminance weighting is off by default.** Weighting each contribution by
`1 / (1 + luma)` is the standard suppression for a single bright sample
dominating the average, and it is not free: down-weighting bright samples
biases the accumulation darker wherever a pixel alternates between a bright and
a dark value, which is exactly what a sub-pixel highlight under jitter does. On
the acceptance scene it cost roughly an order of magnitude of converged
accuracy against the integrated reference. It remains available for
scene-referred HDR input, where one stray firefly is the larger problem.

### Flicker-relaxed rectification

A neighbourhood clamp cannot preserve a feature narrower than a pixel. Half the
frames, a jittered sample misses the feature entirely, its whole neighbourhood
agrees that nothing is there, and the converged history — which is the correct
partial coverage — is clamped away. The accumulation then sits below the truth
and the feature reads as a dim flicker. This is the artifact that makes
temporal anti-aliasing lose thin geometry, and on the acceptance scene it was
the single largest source of both measured error and measured contamination.

The information that separates that case from a genuine cut is not in the
current frame. It is in how the pixel has been disagreeing. The resolver
therefore carries two more reprojected buffers: a running mean of the signed
difference between the input and its reprojected history, and a running mean of
that difference's magnitude. A cut disagrees hard and in one direction, so both
rise together and their difference stays near zero. Sub-pixel flicker disagrees
in alternating directions, so the magnitude rises while the signed mean does
not. Their difference is the oscillation amplitude, and the clip box is widened
by `flickerRelaxation` multiples of it.

A cut is still rejected because the drift is read before the current frame's
difference is folded into it: at the moment a cut arrives, the pixel's record
of disagreement is still empty. Rejected pixels reset their drift along with
their history, and a reset clears both. The two buffers are double-buffered so
the job-parallel path never reads a pixel another row has already written,
which keeps the byte-identical guarantee.

The default of 2 is the measured knee. At that setting the resolve recovers
essentially all of the converged accuracy the clamp costs, and the edge
statistics of the result land on those of the integrated reference rather than
below them — the difference between reconstructing an image and blurring one.

## Measuring instead of inspecting

The roadmap's bar is that temporal anti-aliasing shows "no ghosting" in a
moving scene. "Visible" is not a test, so `post_process_quality.h` turns the
artifacts into numbers the way `shadow_quality.h` did for Phase 28:
`MeasureImageDifference`, `MeasureGhosting`, `MeasureTemporalStability`, and
`MeasureEdgeAliasing`. Each reads borrowed images, allocates nothing, and is
deterministic, and each has unit tests whose answers can be written down before
they are run.

Ghosting is defined against three images of the same scene: the resolve, the
frame with no temporal history, and the frame before it with no temporal
history. For every pixel the two references disagree on, the resolved colour is
projected onto the line from the correct answer to the stale one.

That projection is reported two ways, and which one to assert on is itself a
design decision. The per-pixel ratio divides by how far its own pixel moved, so
a pixel that barely cleared the change threshold turns a negligible error into
a large ratio, and an unweighted mean of those ratios is dominated by pixels
carrying almost no visible signal. `weightedContamination` instead fits one
scalar over the whole image — the least-squares `a` in
`resolved = current + a * (previous - current)` — which weights each pixel by
how much it actually moved. That is the number the acceptance suite asserts on;
the per-pixel mean is reported beside it.

The threshold on it is absolute rather than a ratio against a reference run,
because the run that would define such a ratio — a frame carrying no history at
all — fits zero by construction. What calibrates an absolute threshold is a
control that must miss it by a wide margin, and every threshold in the suite is
paired with one.

One result from that suite is worth stating plainly, because it inverts the
expected trade. Widening the clip box is the direction that trusts history
more, and it *reduces* measured contamination monotonically. A strict clamp
does not keep stale colour out; it throws live colour away, and the hole it
leaves lies along the same axis a trail would, because that is where the
previous frame's detail was. The suite asserts that monotonicity rather than
assuming the opposite.

## Known limits

- All of this is CPU-side. The timings are honest CPU wall time for a software
  renderer and are not a frame budget for a GPU implementation.
- Ambient occlusion is a hemisphere-sampled depth-and-normal estimator, not
  GTAO. Depth of field is a single-layer gather with a bounded circle of
  confusion and does not model occlusion between the layers it blurs across.
- Motion blur follows the velocity buffer alone, so it cannot reconstruct what
  a moving object uncovers.
- The acceptance scene is analytic rather than rasterized. That is what makes
  an exact ground truth, an exact velocity buffer, and an exact one-sample
  render available at once; it also means the measurements carry no rasterizer
  error, and a rasterized scene would measure worse. It also contains no
  occlusion, so the artifact the depth test exists to prevent cannot appear in
  it — that case is covered by unit tests and by the sample instead. A scene
  that measures a renderer is not the same thing as a scene that exercises it,
  and this phase needed both.
- A panning camera's leading edge has no history by construction, and the
  measurements include it. It is the largest single contributor to the
  residual contamination the suite reports.
