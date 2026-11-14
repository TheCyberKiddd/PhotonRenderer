# GPU clustered lighting sample

This D3D12 compute sample builds conservative per-cluster light lists on the
GPU, consumes them in a GGX PBR compute pass, reads back the result, and checks
representative pixels against the CPU PBR implementation. It also compares
GPU-generated cluster membership against the CPU list-builder oracle.

The sample uses a headless, synthetic surface grid rather than a GPU geometry
render pass: its surface records are the same inputs a future G-buffer or
forward raster pass would provide. It is an executable proof of the compute
list-builder and lighting consumer, not an automatic switch in the renderer
plugin or a complete scene-rendering backend.

Build and run:

```powershell
cmake --preset release -B build\release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
cmake --build build\release --target cyber_sample_clustered_lighting
build\release\samples\09_clustered_lighting\cyber_sample_clustered_lighting.exe
```

The sample requires a D3D12 adapter with compute support. On unsupported
platforms, use the CPU clustered-forward path described in the [Phase 27
report](../../docs/reports/phase-27.md). Its HLSL resource bindings are
explicitly matched by the sample's RHI layouts; the general shader DSL does
not yet express structured-buffer loops.
