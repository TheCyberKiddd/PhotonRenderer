# 12_post_process — the post chain, priced pass by pass

Renders a scene with something for every post-processing pass to act on, runs
the chain, and prints what each pass cost and what switching it off changed.
Then resolves the same moving scene three ways — unfiltered, FXAA, and
temporal — and reports the measured edge statistics of each.

This is the Phase 30 bar made visible: every effect toggles at runtime with its
time shown, and the anti-aliasing paths are compared on a scene that moves.

The scene is a floor and a back wall meeting in a crease for ambient occlusion,
a row of posts receding past the focal plane for depth of field, a screen of
thin vertical slats for the anti-aliasing to resolve, and one small surface
driven far above the tone mapper's shoulder for bloom. The camera pans, so the
velocity buffer is non-zero and motion blur and the temporal resolve both have
something to follow.

## Build and run

From the repository root:

```powershell
cmake --preset release
cmake --build --preset release --target cyber_sample_post_process
.\build\release\samples\12_post_process\cyber_sample_post_process.exe
```

The sample writes four images into the current directory. Pass a directory
prefix as the first argument to write them elsewhere:

```powershell
.\build\release\samples\12_post_process\cyber_sample_post_process.exe C:\temp\
```

| File | What it shows |
|---|---|
| `post_process.ppm` | The frame with the whole chain applied |
| `aa_none.ppm` | One sample per pixel, no filter |
| `aa_fxaa.ppm` | The same frame through the edge filter |
| `aa_temporal.ppm` | 24 jittered frames accumulated and reprojected |

## Reading the output

The times are CPU milliseconds. This renderer shades on the CPU, so they are
the whole cost of each effect rather than a GPU queue timestamp, and they are
measured with the job system switched off so the numbers are reproducible
rather than fast. Ambient occlusion and depth of field dominate because both
gather many taps per pixel; the rest of the chain is a single pass over the
image.

Tone mapping and colour grading report the same time because they share one
resolve pass — applying a baked lookup table costs nothing extra once the pixel
is already in registers — and the measured cost of that pass is split evenly
between whichever of the two ran, so the parts still sum to the whole.

The `changed` column is the fraction of pixels that differ from the all-on
frame by more than one 8-bit step when that one effect is switched off. A
toggle that changed nothing would be a checkbox rather than an effect, which is
why the number is printed.

`temporal history` reports what the resolve did with the previous frame: how
many pixels reused history, how many dropped it as a disocclusion and how many
of those were caught by the depth test, how many had history pulled back into
the current frame's neighbourhood, and how many had that neighbourhood widened
because the pixel has been oscillating — the mechanism that keeps the thin
slats from being clamped away.

The depth count is worth watching. This sample is what found the artifact it
prevents: with only colour and velocity, the slats leave visible trails across
the floor as the camera pans, because a floor pixel the slats were covering
carries the floor's velocity and reprojects straight onto the slat. Supplying
the depth buffer drops history on those pixels and the trails go. See the
[Phase 30 design](../../docs/design/phase-30-post-processing.md) for the
mechanism behind both numbers.
