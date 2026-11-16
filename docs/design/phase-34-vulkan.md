# Phase 34 — Vulkan backend

Status: **complete on Windows.** Every sample and golden-image test passes on D3D12 and
Vulkan with the same renderer code, verified on an Intel HD Graphics 530 (Vulkan 1.3):
190 of 190 CTest tests pass, including a `_vulkan` run of every renderer GPU test, both
backends' run of every RHI sample, and the backend comparison against the software
reference. The Vulkan backend compiles the renderer's HLSL itself
([ADR 0030](../adr/0030-vulkan-hlsl-compiler-and-d3d12-semantics.md)), and
[ADR 0029](../adr/0029-vulkan-backend.md) records the backend's founding decisions.
Two named items remain (below): a run with the Khronos validation layer, and the
Linux and Android window backends that Vulkan surfaces need.

## Running anything on either backend

One binary serves both backends. The backend comes from `--backend=d3d12|vulkan`, then
the `CYBER_RHI_BACKEND` environment variable, then the program's default
(`cyber/rhi/backend_select.h`).

- Renderer GPU tests, `rhi_timestamps` and `backend_compare` are registered twice:
  `<name>` (D3D12) and `<name>_vulkan` (`cyber_add_vulkan_variant` in
  `cmake/CyberModule.cmake`).
- RHI samples are registered as `sample_<name>_d3d12` and `sample_<name>_vulkan`
  (`samples/CMakeLists.txt`). A sample exits with 77 when the machine has no adapter or
  window system for a backend, which CTest reports as skipped, so a machine without a
  GPU skips rather than fails.
- `rhi_backend_gpu_parity` runs one body on both backends in one process and compares
  the outputs byte for byte.

## What exists

Everything lives in `engines/cyber_rhi/src/vulkan/`.

| Piece | File | Test |
|---|---|---|
| API subset: types, enums, function tables | `vk_api.h`, `vk_api_ext.h` | every Vulkan test |
| Loader, instance, adapters, 1.3 feature gate, optional features, limits, queues, device, validation messenger, naming | `vulkan_context.*` | `rhi_vulkan_context` (prints the optional features and limits) |
| Memory sub-allocation, raw tile allocations, sparse images | `vulkan_memory.*` | `rhi_vulkan_memory` |
| Command pool, `vkQueueSubmit2`, timeline fences | `vulkan_commands.*` | `rhi_vulkan_commands` |
| RHI state to stage, access and layout | `vulkan_barriers.h` | `rhi_vulkan_commands` |
| Set layouts (update-after-bind arrays), pipeline layouts, pipelines | `vulkan_pipelines.*` | `rhi_vulkan_pipelines` |
| HLSL to SPIR-V compiler | `hlsl_spirv*.{h,cpp}` | `rhi_hlsl_spirv` (no GPU), and every Vulkan run |
| Win32 surface, FIFO swapchain | `vulkan_swapchain.*` | `rhi_vulkan_swapchain`, `rhi_backend_gpu_parity` |
| RHI device, resource half; zero-filled creation | `vulkan_device.cpp` | `rhi_backend_parity` |
| RHI device, command half; per-pipeline HLSL compilation | `vulkan_device_gpu.cpp` | `rhi_backend_gpu_parity`, renderer `_vulkan` tests |
| Timestamps, pipeline caches, reserved textures | `vulkan_device_features.cpp` | `rhi_timestamps_vulkan`, `rhi_backend_gpu_parity`, `renderer_gpu_texture_pool_vulkan`, `renderer_gpu_streaming_render_vulkan` |

## The HLSL compiler

The renderer's shaders are HLSL. D3D12 compiles `ShaderBytecodeFormat::HlslSource` with
d3dcompiler; Vulkan compiles the same source with the backend's own compiler to SPIR-V
1.5 (ADR 0030). SPIR-V given directly is still accepted.

- **When it compiles.** `CreateShader` compiles once with a provisional binding layout,
  so syntax and type errors surface there with `name:line:col`. Each pipeline compiles
  again against its layout and discards the module once the pipeline exists.
- **Registers to bindings.** Register space N is bind group slot N. The register of a
  layout entry is assigned as the D3D12 root signature assigns it: an explicit
  `shaderRegister`, or the running count of earlier entries in its class (b; t for
  sampled textures and read-only buffers; u; s). Declarations that land in one
  binding share one descriptor array. `b0, space0` is the push-constant block when the
  layout has push constants.
- **Stage linking.** Vertex outputs are numbered in declaration order and pixel inputs
  are found by semantic, so a pixel shader may read a subset in any order.
  Integer varyings are flat. Vertex inputs are `TEXCOORD<n>` at location n, the same
  rule the D3D12 input layout uses.
- **Meaning kept from HLSL and D3D12.** HLSL `floatRxC` is a SPIR-V matrix whose columns
  are the HLSL rows, so `m[i]`, constructors and `mul` need no transposes. cbuffers use
  16-byte register packing and structured buffers tight packing, as explicit offsets
  (scalar block layout, required of the device). Float-to-integer conversion saturates with NaN to 0.
  `SV_Position.w` in a pixel shader is clip w. `SV_InstanceID` starts at 0
  (InstanceIndex - BaseInstance, which needs shaderDrawParameters; a shader reading it is refused without). Scalar `&&`, `||` and `?:` evaluate only what they
  need. `precise` becomes `NoContraction`. Shift amounts use their low five bits.
- **Refusals.** Unsupported constructs (switch, preprocessor macros, geometry stages,
  resources as parameters, 16/64-bit types, row_major) fail with "not supported by the
  Vulkan HLSL compiler" and a location. Hostile input is bounded (nesting depth,
  integer ranges) and fuzzed.

## The RHI device on Vulkan

| RHI concept | Vulkan mapping |
|---|---|
| Bind group slot N, entry `binding` B | descriptor set N, binding B; unused slots get an empty set layout |
| Binding array | update-after-bind and partially bound, from update-after-bind pools (bindless); a layout with dynamic-offset buffers keeps ordinary bindings |
| Push constants | `vkCmdPushConstants`; HLSL `b0, space0` |
| Viewport | negative height, so clip-space y and winding keep their D3D12 meaning |
| Barriers | derived from the RHI `from` and `to` states |
| Render pass | dynamic rendering |
| Back buffers | start in `Present`; acquire and present hide the binary semaphores |
| New resources | zero-filled (clear or fill in one batch ahead of the next submission), as D3D12 committed resources start zeroed |
| Out-of-range shader access | reads zero, writes dropped (`robustBufferAccess`, `robustImageAccess`) |
| Fences | timeline semaphores; each queue kind also signals an internal timeline that orders sparse binds |
| Timestamps | query pool plus a mapped readback buffer; resolve copies, then resets the queries |
| Pipeline caches | `VkPipelineCache`; foreign or stale bytes give an empty cache |
| Reserved textures | sparse-resident images in the standard 64 KiB block shape; tile heaps are raw allocations; `UpdateTileMappings` is `vkQueueBindSparse` ordered on its queue |
| Wireframe, depth clip off | `fillModeNonSolid`, `depthClamp` |
| Limits | read from the device (offset alignments, attachments, vertex bindings, descriptor-indexing array limits) |

## What the cross-backend tests prove (HD Graphics 530)

- One HLSL compute shader through a bind group and push constants gives the same 1000
  words on D3D12 and Vulkan; one HLSL triangle gives the same picture.
- Both backends present swapchain frames (the D3D12 swapchain now creates its buffers in
  the linear twin of an sRGB format, as flip-model swapchains require).
- Pipeline-cache bytes round-trip on both backends; timestamps resolve and reuse.
- GPU culling of a million instances matches the CPU reference exactly on Vulkan (0
  disagreements), as does Hi-Z occlusion; virtual-texture sampling and feedback match
  the CPU sampler; BC6H decodes match the hardware decoder; reserved-texture streaming
  reads 92 160 samples with none wrong.
- The clustered-lighting sample's GPU PBR matches the CPU (max error 0) on Vulkan;
  every other RHI sample runs on both backends.
- `backend_compare`'s triangle and textured cube match the software reference on
  Vulkan (0 pixels over tolerance).

## Still open

1. **A run with the Khronos validation layer.** The layer is not installed on the
   development machine, so the "zero validation errors" checks pass vacuously here. The
   code enables it with synchronization validation when present and tests require it
   then. One run on a machine with the layer is owed.
2. **Linux and Android surfaces.** `cyber_platform` has no Linux or Android window
   backend yet, so there is no window to make a Vulkan surface for. Without one, the
   instance is created headless and swapchain creation answers `Unsupported` by name;
   everything else (compute, offscreen rendering, every test that needs no window)
   builds and runs on Linux. The Linux build was reviewed by reading, not compiled.

## Behaviour differences to keep in view

| Topic | D3D12 | Vulkan |
|---|---|---|
| Adapter list | DXGI factory, WARP last | instance + physical devices, software last |
| Backend version value | `1200` | `103` (major * 100 + minor) |
| Device-local memory on an integrated GPU | dedicated reported separately | one device-local heap that is system memory |
| Fence | native `ID3D12Fence` | timeline semaphore (required feature) |
| Render state | render targets set on the list | dynamic rendering, no render pass objects |
| Barriers | resource states | stage, access and layout derived from the one state |
| Naming views and samplers | not nameable (descriptors) | nameable, labels read back |
| Missing GPU or driver | WARP always exists | zero adapters |
| Shaders | HLSL compiled at `CreateShader` | HLSL checked at `CreateShader`, compiled per pipeline; or SPIR-V |
| Storage-buffer offset alignment | 4 bytes | the device's (64 on the HD 530) |
| Ordinary per-stage descriptor limit | descriptor tables | 200 on the HD 530, so arrays are update-after-bind |
| Storage images read without a format | typed UAV loads | needs `shaderStorageImageReadWithoutFormat` (absent on the HD 530; writes work) |
| Unmapped reserved tiles | read zero (tier 2) | read zero when the device has sparse aliasing (they alias one zeroed tile), otherwise unbound |

## Lessons the HD 530 taught

- Creating a Vulkan instance costs about 300 ms on this driver; adapter enumeration is a
  startup cost.
- The driver's DLL initialisation runs inside the first instance creation on the
  caller's stack. A sample whose `main` kept 120 KB of arrays on the 1 MiB main-thread
  stack made it fail with error 1114 and reported zero adapters; the arrays are static.
- An image placed in memory freed by another image inherited stale compression
  metadata and hung the GPU when unwritten texels were sampled. Zero-filled creation
  fixed it, and is what D3D12 guarantees anyway.
