# Phase 27 implementation report

Status: CPU clustered-forward path and standalone D3D12 compute proof validated.
Renderer-plugin integration, automatic backend switching, a rendered heat-map
overlay, and the deferred comparison are not implemented.
Date: 2026-11-08

## Delivered

- `cyber_renderer` builds checked, deterministic CSR light lists with reusable
  allocator-backed scratch storage. `BuildClusterHeatMap` exports maximum
  per-tile occupancy for visualization.
- `cyber_software_renderer` converts point-light bounds and consumes those
  lists in its CPU PBR rasterizer. Invalid/out-of-grid fragments or
  incompatible indices safely use the full local-light list.
- The 1,000-light differential test compares clustered PBR against brute force
  and checks eight representative pixels from the actual rasterized image.
- `cyber_software_clustered_pbr_bench` builds moving 1,000-light lists and
  renders a fullscreen scene. Each timed frame includes light-bound
  conversion, list construction, target clear, and rasterized PBR shading; it
  excludes startup, presentation, and GPU work. It verifies full pixel
  coverage and reports configuration, occupancy, median, and p95.
- CPU-only means using the software renderer; this implementation does not
  auto-detect GPU capability or transparently replace a GPU renderer.
- `cyber_sample_clustered_lighting` is a standalone, headless D3D12 compute
  proof. It builds fixed-capacity GPU cluster lists, shades a synthetic surface
  grid with GGX PBR, checks GPU list membership against the CPU cluster builder,
  and compares representative shaded pixels against `PbrPixelShader`. It does
  not rasterize geometry, consume a real G-buffer, integrate with the renderer
  plugin, or implement automatic backend fallback.

## Validation

Host: Windows 10 Pro, Intel Core i5-6400, 4 logical processors, MSVC 19.44.

- `cmake --build build\phase27-debug --target cyber_test_software_clustered_pbr` — passed.
- `ctest --test-dir build\phase27-debug -R "^software_clustered_pbr$" --output-on-failure` — passed, 1/1.
- `ctest --test-dir build\phase27-debug --output-on-failure` — passed, 128/128.
- `cmake --build build\phase27-release --target cyber_test_software_clustered_pbr cyber_software_clustered_pbr_bench` — passed.
- `ctest --test-dir build\phase27-release -R "^software_clustered_pbr$" --output-on-failure` — passed, 1/1.
- `build\phase27-release\engines\cyber_software_renderer\cyber_software_clustered_pbr_bench.exe` — passed; 57,600/57,600 pixels rendered, 5,760 clusters, 143,811 indices, maximum occupancy 84.
- `py -3.14 scripts\check_structure.py` — passed; 17 modules and 1 contract.
- `py -3.14 scripts\layer_check.py` — passed; 17 modules.
- Targeted `clang-format --dry-run --Werror --style=file` — passed.
- `py -3.14 scripts\check_format.py` with clang-format 18.1.8 — passed, 406 files.
- `git diff --check` — passed.

## D3D12 compute validation

The sample was built and run on the Windows 10 Pro machine listed above, using
its Intel HD Graphics 530 D3D12 adapter. This is one adapter's correctness
smoke test, not a hardware-compatibility matrix or a GPU performance benchmark.

- `cmake --build build\phase27-release --target cyber_sample_clustered_lighting` — passed.
- `build\phase27-release\samples\09_clustered_lighting\cyber_sample_clustered_lighting.exe` — passed; 1,000 lights, 320x180 synthetic surfaces, 5,760 clusters, 143,962 GPU indices, maximum occupancy 84, zero overflow, and 12 representative PBR pixels validated. Maximum reported RGB error was `0.000000` to six decimal places, below the `2e-3` acceptance threshold.
- Every CPU-conservative light was present in its GPU cluster list. Any overflow would fail the run; shading has an all-lights fallback for overflowed clusters.

The sample uses handwritten HLSL and explicitly declared RHI layouts because
the current shader DSL does not express the structured-buffer loops required
by these kernels. This prototype does not change the renderer's reflection
contract.

## CPU frame measurement

The Release benchmark used 320x180, 16-pixel clusters, 24 depth slices,
1,000 moving lights, three worker threads, five warmup frames, and 101 measured
frames. Four consecutive runs produced:

| Run | Median frame time | p95 frame time |
|-----|-------------------|----------------|
| 1   | 32.836 ms         | 73.607 ms      |
| 2   | 36.015 ms         | 57.556 ms      |
| 3   | 32.057 ms         | 48.512 ms      |
| 4   | 29.752 ms         | 56.821 ms      |

This CPU-only path is correct but does **not** meet a 16.67 ms / 60-fps budget,
even at the benchmark's low resolution. Performance varies materially between
runs and is a single-machine measurement, not a hardware-matrix claim. The
CPU path is therefore reported as a correctness-tested option rather than a
real-time performance claim.

## Explicitly deferred

The compute kernels remain an isolated D3D12 sample rather than a production
renderer path. The current shader DSL/reflection path does not express the
structured-buffer loops used by the sample, and the sample's surface records
are synthetic rather than generated by GPU rasterization. The exported
occupancy data is not itself a rendered debug overlay; automatic backend
selection, a real G-buffer/forward-renderer integration, deferred rendering,
and transparent draw integration are also outside this deliverable.
