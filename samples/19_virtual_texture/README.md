# 19_virtual_texture — virtual texture with a bounded page cache

This experimental CPU-reference sample renders a terrain backed by a
65,536-by-65,536 virtual texture while keeping only a 24-by-24 cache of
128-texel pages resident. Software pixel shading requests the pages it needs;
the virtual-texture system loads pages in the background and resolves missing
pages through coarser resident mips.

The sample reports cache uploads and evictions, frame-update time, and the
fraction of sampled pixels rendered at the requested mip or within one mip.
It writes the rendered frame, cache atlas, and page-residency map as PPM files.
This demonstrates CPU-side page feedback and residency, not D3D12 reserved
resources or hardware sampler feedback.

## Build and run

From the repository root in PowerShell:

```powershell
cmake --preset release
cmake --build --preset release --target cyber_sample_virtual_texture
.\build\release\samples\19_virtual_texture\cyber_sample_virtual_texture.exe --frames 900
```

The optional `--frames N` argument changes the run length. An optional output
prefix can be supplied as the first argument.
