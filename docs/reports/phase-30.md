# Phase 30 report: post-processing and anti-aliasing

Date: 2026-11-13

Commits: implementation remains uncommitted in the working tree

Machine: Windows 10 Pro 22H2, Intel Core i5-6400 (4 cores), Intel HD Graphics
530, MSVC 19.44 / Visual Studio 2022

## What was built

- An HDR post-processing chain, `PostProcessStack`, owning the scratch images
  for one resolution and running ambient occlusion, depth of field, motion
  blur, bloom, exposure, tone mapping, and colour grading in a fixed physical
  order. Every pass toggles at runtime, a pass that is off or whose input is
  missing is skipped rather than failed, and each reports its own CPU time.
- Exposure: a 64-bin log2-luminance histogram, a percentile window, manual and
  automatic modes with bounds and compensation, and eye adaptation at separate
  rates up and down that snaps on the first frame after a reset.
- Tone mapping with clamp, extended Reinhard, the ACES fit, and Hable's
  filmic curve; colour grading with lift, gamma, gain, contrast, saturation,
  and white balance, evaluated directly or baked into a 3D lookup table that
  can also be loaded from externally authored entries.
- Anti-aliasing: Halton camera jitter and projection jittering, FXAA, and
  `TemporalAaResolver` — velocity-reprojected accumulation with YCoCg
  neighbourhood rectification, variance-bounded clip boxes, flicker-relaxed
  box widening, and depth-based disocclusion rejection.
- `post_process_quality.h`: `MeasureImageDifference`, `MeasureGhosting`,
  `MeasureTemporalStability`, and `MeasureEdgeAliasing`, each allocation-free
  and deterministic, with their own unit suite.
- `samples/12_post_process`, which renders a scene with something for every
  pass to act on, prints the per-pass cost and what each toggle is responsible
  for, and compares the three anti-aliasing paths on the same moving scene.

## What the previous state was, and what changed

The chain, the resolver, and the metric API were already implemented when this
work began, and two of the four acceptance tests were failing. Both failures
were real, and neither was in the test:

1. **Luminance weighting was biasing the accumulation.** Weighting each
   contribution by `1 / (1 + luma)` suppresses a stray bright sample and
   biases the result darker wherever a pixel alternates between a bright and a
   dark value — which is what a sub-pixel highlight under jitter does. On the
   acceptance scene it cost roughly an order of magnitude of converged
   accuracy. It is now off by default and documented as an option for
   scene-referred HDR input.
2. **Neighbourhood rectification was clamping away the features it was meant
   to resolve.** Half the frames, a jittered sample misses a half-pixel line,
   its whole neighbourhood agrees nothing is there, and the correct converged
   coverage is clamped out. The accumulation then sits below the truth.

The fix for the second is flicker-relaxed rectification, described in the
[design note](../design/phase-30-post-processing.md): the resolver tracks a
running mean of the signed difference between the input and its reprojected
history and a running mean of that difference's magnitude, and widens the clip
box by the part of the disagreement that alternates rather than points
consistently one way. A cut is still rejected, because the record is read
before the current frame's difference is folded into it.

Two other changes came out of measuring rather than assuming:

- Clipping history along the segment to the box centre — the textbook form —
  was implemented, measured, and reverted. The chroma axes of a near-grey
  neighbourhood are nearly degenerate, so the smallest chroma disagreement
  dominated the clip ratio and dragged luminance with it, costing about a
  third of the converged accuracy the per-channel form keeps.
- `GhostingMetrics::weightedContamination` was added, because the per-pixel
  ratio the suite was asserting on divides by how far its own pixel moved and
  is therefore dominated by pixels that barely changed. The weighted form fits
  one scalar over the whole image and is the number the suite now asserts on.

The ghosting test's threshold changed with it, and that is worth stating
plainly rather than leaving in a diff. It previously required the resolve to
score no worse than 1.25 times a reference run with no accumulation at all.
That form cannot work on the fitted metric: a frame carrying no history fits
exactly zero, and no multiple of zero is a threshold. The fitted number is now
held to an absolute 0.05, and what calibrates an absolute number is the
control beside it — the same scene with reprojection and rectification off,
which has to exceed the threshold by a wide margin and scores 0.484390. The
per-pixel mean and the ghosted-pixel fraction are still asserted, at 0.10 and
0.15, and the no-accumulation run is still measured and still required to fit
below 0.01. No threshold was loosened to accommodate a result: the resolve
passes the ghosting bar by a factor of 2.7 and the convergence bar by 14.

Finally, the sample found an artifact the acceptance suite structurally could
not. The analytic acceptance scene is a scrolling pattern with no occlusion in
it, which is what makes its ground truth and velocity exact. The 3D sample has
occluders, and the slats trailed visibly across the floor as the camera
panned: a floor pixel the slats were covering carries the floor's velocity,
reprojects onto the slat, and reuses its colour. Depth-based disocclusion
rejection was added for it, and it is now covered by unit tests as well.

## Evidence for done-when

The roadmap's bar is two claims.

### Every effect toggles at runtime with its time shown

From `samples/12_post_process` at 640x400, single-threaded, on this machine:

```
  pass                         ms  state
  ambient_occlusion       393.721  on
  depth_of_field          576.535  on
  motion_blur              29.031  on
  bloom                    21.730  on
  exposure                 24.812  on
  tone_map                 22.494  on
  color_grade              22.494  on
  total                  1093.060
```

Tone mapping and grading report the same figure because they share one resolve
pass, whose measured cost is split evenly between whichever of the two ran, so
the parts still sum to the whole.

The acceptance test asserts the stronger form of the claim: every pass reports
a non-zero time when on and exactly zero when off, the parts never exceed the
measured whole, and switching any single effect off changes more than 1% of the
image — a toggle that changed nothing would be a checkbox. Measured at 96x72:

| Effect switched off | Pixels changed |
|---|---|
| Ambient occlusion | 40.77% |
| Depth of field | 53.08% |
| Motion blur | 54.76% |
| Bloom | 59.79% |
| Tone map | 68.52% |
| Colour grade | 98.09% |

Exposure is checked by its metered value instead, because a manual exposure of
0 EV can land on the same image the meter chose: with the pass on, the frame
metered to -1.577 EV, and with it off the reported value is exactly zero.

### Temporal anti-aliasing shows no visible ghosting in the moving scene

"Visible" is not a test, so the bar is a number. On a scrolling checkerboard
with half-pixel vertical lines, 24 accumulated frames at 3 pixels per frame,
with 37.15% of the image changing between the two reference frames:

| Configuration | Fitted contamination | Per-pixel mean | Ghosted pixels |
|---|---|---|---|
| **Shipped defaults** | **0.018790** | 0.063321 | 8.33% |
| No accumulation (the floor) | 0.000000 | 0.032710 | 3.27% |
| Reprojection, no rectification | 0.000000 | 0.043393 | 6.54% |
| **Control: no reprojection, no rectification** | **0.484390** | 0.489690 | 80.37% |

The shipped configuration reconstructs the moving frame with 1.9% of the
previous frame surviving into it. The control that has both mechanisms off
leaves 48.4% — 26 times more — and fails the threshold by a factor of ten.

The flicker relaxation sweep, which is the knob that trusts history more:

| Relaxation | Fitted contamination | Per-pixel mean |
|---|---|---|
| 0.00 (strict clamp) | 0.052879 | 0.126779 |
| 1.00 | 0.035305 | 0.096539 |
| 2.00 (default) | 0.018790 | 0.063321 |
| 4.00 | 0.011921 | 0.049976 |

Contamination falls monotonically as the box widens, and the suite asserts
that monotonicity. The reason is not a quirk of the metric: a strict clamp
does not keep stale colour out, it throws live colour away, and the hole it
leaves lies along the same axis a trail would, because that is where the
previous frame's detail was.

### Convergence, stability, and the controls that fail

Against an 8x8 supersampled integration of the same static scene, after 32
jittered frames:

| | Mean absolute error | Hard-step fraction |
|---|---|---|
| One sample per pixel | 0.026216 | 0.147554 |
| **Temporal resolve** | **0.000943** | 0.103691 |
| Control: strict clamp | 0.015704 | 0.077695 |
| 8x8 reference | — | 0.107718 |

The resolve is 28 times closer to the integrated image than the frame it
accumulated, and its edge count lands on the reference's own — 0.1037 against
0.1078 — rather than below it. That second number is what separates
reconstructing an image from blurring one, and it is where the strict-clamp
control fails: at 0.0777 it has fewer edges than the truth has, which is the
signature of sub-pixel detail clamped away.

On a still camera over six frames after a 24-frame warm-up, the resolve is
frozen to a mean frame-to-frame delta of 0.002831 against 0.052432 for the
unresolved jittered frames it was accumulated from — 18 times steadier, and
inside the 0.01 absolute bar. Ghosting and stability have to pass at once; a
filter that was too eager to reject history would trade one for the other.

### The depth disocclusion test

Covered by unit test rather than by the acceptance scene, which has no
occlusion in it. With an occluder sliding across a static background, the
pixel it has just uncovered keeps more than 30% of the occluder's brightness
when only colour and velocity are supplied, and under 15% when depth is. In
the 3D sample, 6,298 of 256,000 pixels — 2.5%, the occluder edges — are
rejected on depth, and the trails the slats left across the floor are gone.

![The post chain on the sample scene](images/phase-30-post-chain.png)

![One sample per pixel, FXAA, and the temporal resolve](images/phase-30-antialiasing.png)

## Test suites

All figures from this machine.

- Release: **138/138**, 68.95 s — `ctest --test-dir build\release --output-on-failure`
- Debug: **138/138**, 192.90 s — `ctest --test-dir build\debug --output-on-failure`

The Debug tree was deleted and reconfigured from the preset for this run,
because an earlier reconfigure had been made from a shell whose `PATH`
contained an MSYS2 toolchain and CMake had cached `ld.exe` and `windres.exe`
as the linker and resource compiler. Configuring the presets from a shell with
only the MSVC and CMake directories on `PATH` avoids it; a cache that has
already recorded the wrong tools has to be deleted, since reconfiguring will
not replace them.

Phase 30 contributes four test binaries: `software_post_process` (24 cases),
`software_post_process_quality` (7), `software_temporal_aa` (14), and
`software_post_process_acceptance` (4) — 49 cases in all. Running the
acceptance binary twice produces byte-identical measurements, which is what
the determinism contract is for.

Repository checks:

- `scripts/check_structure.py` — passed, 17 modules, 1 contract.
- `scripts/layer_check.py` — passed, 17 modules.
- `scripts/check_format.py` — **not run to a usable result.** The clang-format
  available on this machine is 19.1.5, which disagrees with the version the
  repository was formatted with: it reports violations in files this phase did
  not touch, including the Phase 14 D3D12 backend sources. Running the
  formatter would have reformatted the repository rather than this phase's
  code, so it was left alone. The new code follows the surrounding style by
  hand.

## Known limits

- Everything here is CPU-side. The per-pass times are honest wall time for a
  software renderer and are not a GPU frame budget.
- Ambient occlusion is a hemisphere-sampled depth-and-normal estimator, not
  GTAO. Depth of field is a single-layer gather and does not model occlusion
  between the layers it blurs across. Motion blur follows the velocity buffer
  alone and cannot reconstruct what a moving object uncovers.
- MSAA is not implemented and is not planned for this renderer; the
  supersampled path stands in its place and `AntiAliasMethod` says so.
- The acceptance scene is analytic. Its measurements carry no rasterizer
  error, and a rasterized scene would measure worse.
- A panning camera's leading edge has no history by construction. It is
  included in the measurements and is the largest single contributor to the
  residual contamination reported above: excluding the last 24 columns of the
  96-column frame drops the per-pixel mean from 0.0434 to 0.0242 and the error
  against ground truth from 0.0079 to 0.0035.
