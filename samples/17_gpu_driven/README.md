# 17_gpu_driven — a million instances, culled on the GPU, one draw call

Describes a 1000 x 1000 voxel landscape — one cube per column, a million
instances — to the Phase 32 GPU-driven renderer once, then renders it. Each
frame the CPU writes the camera and records three things: a compute dispatch
that frustum-culls every instance and picks one of three LODs for it, a
compute dispatch that compacts the surviving batches into indirect draw
arguments, and one `DrawIndexedIndirectCount`. The GPU decides which cubes
are in view, which mesh each one draws — a cube, an octahedron, or a
tetrahedron, by projected size — and how many indirect draws there are.

The LOD thresholds are 1.5 and 0.6 pixels of projected radius, so the reduced
meshes appear only where a voxel is already about a pixel across.

This is the Phase 32 bar made visible: a million objects culled on the GPU and
drawn with a handful of calls, at a CPU cost that does not depend on how many
there are.

## Build and run

From the repository root:

```powershell
cmake --build --preset release --target cyber_sample_gpu_driven
.\build\release\samples\17_gpu_driven\cyber_sample_gpu_driven.exe
```

The sample needs a Direct3D 12 adapter and prefers a hardware one over WARP.
It writes `gpu_driven.ppm` to the current directory.

## Expected result

A sunlit voxel landscape of grass, rock, sand, water, and snow, fading into
haze toward the horizon, and a summary like this one from an Intel HD Graphics
530:

```text
adapter:   Intel(R) HD Graphics 530
instances: 1000000 described, 641518 in view (358482 frustum-culled)
LODs:      102202 cubes, 441853 octahedra, 97463 tetrahedra
commands:  2 culling dispatches, 1 draw call carrying 3 indirect draws
CPU:       0.549 ms median to record culling and the draw
frame:     51.09 ms median submit to completion
wrote gpu_driven.ppm
```

The CPU figure is the time to record culling and the draw, which stays the
same whatever the instance count; the frame figure is the GPU's, dominated on
this adapter by the five million triangles those LODs submit.
