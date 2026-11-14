# Samples

Each sample is runnable, built by the CMake presets, and covered by a test once
the relevant phase exists. A sample that is not tested will break.

| Folder | Purpose | Phase |
|---|---|---|
| `00_window/` | Visible platform window with resize, DPI, and fullscreen events. | 5 |
| `01_gradient/` | Animated software gradient, 60 Hz pacing, and the frame-time overlay. | 6 |
| `02_raster/` | Tile-parallel software triangle rasterizer with interpolated colors. | 7 |
| `03_depth/` | Depth-tested cube over a perspective floor, with orbit and fly cameras. | 8 |
| `04_textured/` | Checker-textured cube with Lambert diffuse and a Blinn-Phong highlight. | 9 |
| `01_triangle/` | First RHI triangle. | 14 |
| `02_cube/` | Textured, rotating cube through the RHI. | 14 |
| `07_materials/` | Reflected material definitions and instances rendered through D3D12. | 18 |
| `08_frame_streaming/` | 10,000 unique-transform draws comparing dynamic offsets and bindless indices. | 19 |
| `04_lighting/` | PBR sphere grid and light types. | 26 |
| `11_reflections/` | Captured room probe lighting a polished sphere; checker floor, scene props, and cast shadows. | 29 |
| `10_antialiasing/` | Renderer-wide supersample-and-reconstruct anti-aliasing, quality and filter comparison. | 28 |
| `12_post_process/` | HDR post chain priced pass by pass, each effect toggled, and the FXAA and temporal paths compared on a moving scene. | 30 |
| `14_global_illumination/` | Cornell-style room baked into an irradiance volume and rendered with and without it, ray-traced lamp shadows, the denoiser on one-sample soft shadows, and the ray-tracing policy's answer for every local adapter. | 31 |
| `15_gallery/` | Rich room of varied shapes and materials, comparing the Phase 31 irradiance bake with image-based lighting. | 31 |
| `17_gpu_driven/` | A million-instance voxel landscape frustum-culled and LOD-selected on the GPU and drawn with one indirect-count draw call. | 32 |
| `18_texture_streaming/` | 400 BC1 textures, 16.7 times the video-memory budget, streamed down a corridor flythrough with hitch, budget, and residency-view measurements. | 33 |
| `19_virtual_texture/` | Experimental software-rendered terrain with virtual page feedback, a bounded physical cache, and visible mip-residency output. | 33 |
| `host_app/` | Stub Cyber Engine host that loads renderer plugins. | 25 |

Each sample has a README with the exact command to build and run it, and the
expected result.
