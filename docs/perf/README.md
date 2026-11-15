# Performance log

Performance claims are recorded here with the machine and the build. The log is
append-only; a new measurement is a new entry, not an edit of history.

## Entry template

```markdown
## YYYY-MM-DD — <change or phase>

- Machine: CPU, GPU, RAM, OS, driver
- Build: preset, compiler version, commit
- Scene: name, resolution, object count
- Metric: median, p95, p99, slowest 1%, GPU time per pass
- Result: number and units
- Notes: what changed, what was measured, what was not
```

## Budgets

Budgets are set per phase and recorded in `engines/<module>/docs/MODULE.md`.
Until measured, mark them "none yet". Do not invent a budget to look rigorous.

Planned budget categories:

- CPU frame time: update, culling, submission.
- GPU time per pass.
- Frame-time consistency: slowest 1% of frames.
- Memory: system and video memory, fragmentation.
- Asset load and cook time.
- Shader compile and cache hit rate.

## Rules

- Benchmark on fixed hardware; compare medians and percentiles, not averages.
- Record the driver version; a GPU result without a driver is incomplete.
- A benchmark that is not in CI will regress. Add it to the benchmark suite when
  the phase introduces it.
- Investigate the three largest costs before optimizing anything else.

## 2026-11-14 — Software renderer performance pass (Phase 33)

- Machine: Intel Core i5-6400 (4 cores), Intel HD Graphics 530, Windows 10 Pro 22H2
- Build: release preset, MSVC 19.44
- Scene: `cyber_software_frame_bench` (36 spheres of 576 triangles and a ground
  plane, 256x256 mipped base colour) and the windowed samples 02-04
- Metric: mean frame time over 40 frames after 8 warm-up frames
- Result: flat 320x180 23.07 -> 9.58 ms; flat 1280x720 30.68 -> 24.10 ms; pbr
  320x180 22.53 -> 13.66 ms; pbr 1280x720 79.17 -> 51.18 ms; pbr + 4 point
  lights 1280x720 171.74 -> 96.95 ms; 576-triangle draw 0.426 -> 0.240 ms;
  1280x720 present pack 7.2-8.1 -> 3.7-4.1 ms; `04_textured` at 1x 38.1 ->
  45.4 FPS
- Notes: causes and fixes are listed in `docs/reports/phase-33.md`. The largest
  single cause was per-draw heap churn: about 0.5 MB of transient arrays
  allocated and freed per draw, which Windows decommitted and page-faulted back
  in every time (29 faults per draw). Frames that shade millions of samples
  remain fill-rate bound.

## 2026-11-14 — Texture streaming (Phase 33)

- Machine: as above
- Build: release preset, MSVC 19.44
- Scene: `samples/18_texture_streaming`, 400 textures of 2048x2048 BC1 (1.07
  GiB) through a 64 MiB budget, 1,800 frames at 60 Hz
- Metric: main-thread `TextureStreamer::Update` time per frame
- Result: mean 0.21 ms, p99 0.65-0.70 ms, max 0.86-1.19 ms over two runs; no
  frame over 2 ms; peak 1024 of 1024 pages, never exceeded
- Notes: with the sample's main thread at normal priority instead of above
  normal, the worst frame was 4.9 ms with no streaming work in it — preemption
  by the IO threads on a 4-core CPU.

## 2026-11-15 — Interactive textured sample and the resolve pass

- Machine: Intel Core i5-6400 (4 cores), Intel HD Graphics 530, Windows 10 Pro 22H2
- Build: release preset, MSVC 19.44
- Scene: `samples/04_textured` window path, 1280x720, textured cube
- Metric: smoothed frame rate over 100 to 500 frames; per-stage mean from the
  sample's own timers
- Result: default (2x per axis) 15.3 FPS before; at a fixed 2x, 17.6 FPS after
  (resolve 23.5 -> 15.1 ms); the default now adapts to 1x and holds 60 FPS
  (clear 1.9, draw 8.1, resolve skipped, pack 3.5, present 2.1 ms); fixed 1x was
  43 FPS with a 9.2 ms resolve before
- Notes: `ResolveSupersampled` made two job-system dispatches per 32-row band
  (46 per frame) and wrote through `SetPixel`; it is now one dispatch of
  independent row blocks that stream through a ring of filtered rows, with a
  direct store for RGBA32F targets. Output is byte-identical (`software_antialias`
  passes). The sample skips the resolve at one sample per pixel and steps the
  sample count down when frame work exceeds 20 ms, back up after 120 frames under
  6 ms with a 600-frame cooldown; `--supersample` pins it. Samples 01 to 03 were
  already at the 60 FPS cap. A fixed 2x render of a 1280x720 window remains fill
  bound at about 17 FPS.
