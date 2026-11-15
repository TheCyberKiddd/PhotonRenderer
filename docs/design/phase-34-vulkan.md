# Phase 34 — Vulkan backend

Status: **the Vulkan backend is complete for the RHI as the renderer uses it, with two named gaps.** Every device entry the renderer calls is implemented on Vulkan: resources, views, samplers, SPIR-V shaders, fences, bind groups, pipeline layouts, graphics and compute pipelines, command lists (render passes, draws, indirect draws, dispatches, copies, clears, barriers, push constants, labels), submission, transitions and swapchains. `rhi_backend_gpu_parity` runs one frame of RHI code on D3D12 and on Vulkan and compares the output byte for byte. The gaps are SPIR-V from the shader compiler and the Khronos validation layer, below. [ADR 0029](../adr/0029-vulkan-backend.md) records the decisions.

## What exists

Everything lives in `engines/cyber_rhi/src/vulkan/`. Each row is covered by a test
that runs on a real Vulkan 1.3 device and skips cleanly on a machine without one.

| Piece | File | Test |
|---|---|---|
| API subset: types, enums, function tables (core, then images, commands, sync2, descriptors, pipelines, dynamic rendering, presentation) | `vk_api.h`, `vk_api_ext.h` | every test below creates a device through it |
| Loader, instance, adapter list, 1.3 feature gate, queue families, logical device, validation messenger, object naming | `vulkan_context.*` | `rhi_vulkan_context` |
| Memory sub-allocation: blocks per (class, linear, memory type) growing 8 to 256 MiB, first-fit free list with merge, dedicated blocks for large requests, driver allocation cap, persistent mapping, `MemoryStats` | `vulkan_memory.*` | `rhi_vulkan_memory` |
| Buffers, images, image views, samplers | `vulkan_memory.*` | `rhi_vulkan_memory`, `rhi_vulkan_commands` |
| Command pool, `vkQueueSubmit2`, timeline semaphore as the RHI fence, immediate submit | `vulkan_commands.*` | `rhi_vulkan_commands` |
| RHI `ResourceState` to stage, access and layout (barrier translation) | `vulkan_barriers.h` | `rhi_vulkan_commands` (pure logic, no device) |
| Descriptor set layouts, update-after-bind bindless layout, pools, pipeline layouts with push constants, compute and graphics pipelines (dynamic rendering, no render pass) | `vulkan_pipelines.*` | `rhi_vulkan_pipelines` |
| SPIR-V module writer with the specification's section order | `spirv_builder.h` | `rhi_vulkan_pipelines` |
| Win32 surface, FIFO swapchain, acquire and render-finished semaphore rings, recreation | `vulkan_swapchain.*` | `rhi_vulkan_swapchain` |
| RHI `DeviceInterface`, resource half: buffers, textures, views, samplers, SPIR-V shaders, timeline fences, queues, capabilities, memory stats, object names, leak report | `vulkan_device.*` | `rhi_backend_parity` (one body on D3D12 and Vulkan) |
| `VulkanBackend()` entry and adapter list | `vulkan_backend.cpp`, `cyber/rhi/vulkan.h` | `rhi_vulkan_context` |

What the hardware tests prove, on an Intel HD Graphics 530 (Vulkan 1.3):

- a 64 KiB buffer and a 64x64 RGBA8 image round-trip bit-exactly through
  device-local memory with barriers derived from RHI states;
- a compute shader reads and writes storage buffers through a descriptor set and
  push constants, with a dispatch that is not a multiple of the workgroup size;
- a graphics pipeline renders a full-screen triangle with dynamic rendering, and
  the corner and centre pixels match the barycentric colours;
- a partially bound, variable-count, update-after-bind sampled-image array is
  created, allocated and written;
- a swapchain presents frames, and the first frame's pixels read back as the clear
  colour;
- every allocation returns to the starting level after teardown.

## The RHI device on Vulkan

`vulkan_device.cpp` holds the resource half and `vulkan_device_gpu.cpp` the command
half; `vulkan_device_internal.h` is the shared state. Conventions that let renderer
code written against D3D12 run unchanged:

| RHI concept | Vulkan mapping |
|---|---|
| Bind group slot N, entry `binding` B | descriptor set N, binding B; dynamic-offset entries use the `*_BUFFER_DYNAMIC` types; unused slots get an empty set layout so set indices stay contiguous |
| Push constants | `vkCmdPushConstants` against the pipeline layout of the bound pipeline |
| Viewport | submitted with a negative height, so clip-space y points up as in D3D12 and front-face winding keeps its D3D12 meaning |
| Barriers | stage, access and layout derived from the RHI `from` and `to` (`vulkan_barriers.h`); tracked state updated as the barrier is recorded |
| Render pass | `vkCmdBeginRendering`; pipelines name their attachment formats, there are no render pass objects |
| `ClearColorTexture` and `ClearDepthStencilTexture` | a one-subresource render pass with a clear load op, valid in the same `RenderTarget` and `DepthWrite` states D3D12 clears in; the transient views live until the list is reset |
| Back buffers | start in `Present` (the images are moved there once at creation); acquire gates the graphics queue on the image, present waits for every earlier graphics submit through an internal timeline, so callers never see a binary semaphore |
| Cross-queue resources | created `CONCURRENT` across the distinct queue families, so no ownership transfers |
| Fences | timeline semaphores; a timed-out wait reports `NotFound`, as D3D12 does |
| Limits | image sizes and push-constant size are read from the device; the rest are the specification's guaranteed minimums |

## What the cross-backend tests prove

`rhi_backend_gpu_parity` records RHI code once and runs it on every compiled-in
hardware backend, with only the shader bytes differing (HLSL for D3D12, SPIR-V from
`SpirvBuilder` for Vulkan):

- a compute dispatch through a bind group of two storage buffers and push constants
  produces the same 1000 words on D3D12 and Vulkan;
- a graphics pass (pipeline layout, pipeline, render pass, viewport, scissor, vertex
  buffer, draw, texture-to-buffer copy) produces the same picture on both within
  interpolation rounding, and the corner and centre pixels match the expected
  barycentric colours;
- a swapchain acquires and presents frames (Vulkan; the D3D12 swapchain is created
  against a hidden window in this test and the D3D12 backend declines it, so that half
  is skipped on D3D12 and covered by the D3D12 swapchain tests);
- no object leaks after teardown on either backend.

`rhi_backend_parity` covers the resource contract (host access rules, labels, stale
handles, fence semantics) identically on both.

## What does not exist yet

1. **SPIR-V from the shader compiler.** The one external dependency left. The Vulkan
   device accepts SPIR-V (`ShaderBytecodeFormat::SpirV`) and `spirv_builder.h` writes
   it, but the compiler (ADR 0019) lowers HLSL to DXBC/DXIL only, so the renderer's own
   shaders cannot run on Vulkan until a SPIR-V producer exists. The Windows SDK's
   `dxc.exe` and `dxcompiler.dll` on the development machine were built without SPIR-V
   code generation ("SPIR-V CodeGen not available"). Closing the gap needs either a
   DXC build with SPIR-V enabled (it compiles the same HLSL with `-spirv`, and the
   binding convention above maps to `-fvk-b-shift`, `-fvk-t-shift` and friends per
   register space), or a SPIR-V back end in the project's own compiler over its IR.
   Either way the device side is ready.
2. **Validation.** The Khronos validation layer is not installed on the test machine,
   so every "zero validation errors" assertion is vacuous here. The code enables the
   layer with synchronization validation when present and the tests require it when
   installed; the first run on a machine with it is still owed.
3. **Timestamp queries, pipeline caches and reserved textures** answer `Unsupported`
   on Vulkan (the capability report omits them).
4. **Linux and Android surfaces** with their window backends.
5. **Every sample and golden-image test on both backends** needs item 1.

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
| Shaders | HLSL source or DXIL | SPIR-V only |

## Performance note

Creating a Vulkan instance on the Intel HD Graphics 530 driver costs about 300 ms,
so adapter enumeration is a startup cost, not a per-frame one.
