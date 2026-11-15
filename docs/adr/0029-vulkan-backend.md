# 0029 — The Vulkan backend: own declarations, runtime loading, and the increment plan

- **Status:** accepted
- **Date:** 2026-11-15
- **Deciders:** @TheCyberKiddd
- **Related:** Phase 34, [design note](../design/phase-34-vulkan.md), ADR 0003
  (integration and packaging), ADR 0013 (RHI object model), ADR 0016 (command
  lists and synchronization), ADR 0017 (D3D12 backend)

## Context

Phase 34 is the second backend and the first test of whether the RHI is really
API-neutral. Three questions have to be answered before code is written, because
each one is expensive to reverse:

1. Do the Vulkan headers and loader count as "driver headers" under the project's
   no-third-party-libraries rule, the way `d3d12.h` does?
2. How does the backend find the Vulkan runtime, and what does a machine without
   one do?
3. Which of the API's competing mechanisms does the backend use for rendering
   state, synchronization and binding?

## Decision

1. **The loader is a driver component; the headers are not taken.** `vulkan-1.dll`
   and `libvulkan.so.1` ship with GPU drivers exactly as `d3d12.dll` ships with
   Windows. The Khronos header repository is a separate dependency with its own
   licence and update cadence, and the project's rule is to avoid those. The
   backend therefore declares the part of the Vulkan 1.3 API it uses in
   `src/vulkan/vk_api.h`, transcribed from the specification, and loads entry
   points at run time. A declaration is added when code needs it. A structure the
   backend only partly reads (`VkPhysicalDeviceProperties`) is declared with an
   opaque tail larger than the real structure, so a driver can never write past it.
2. **Nothing links against Vulkan.** The loader is opened with `LoadLibrary` or
   `dlopen` each time it is needed; the operating system reference-counts the
   module, so the backend keeps no global. A build therefore compiles the backend on
   a machine with no SDK and no driver, and a machine with no runtime reports zero
   adapters instead of failing to start. `IsVulkanCompiledIn()` is a build
   constant; whether a device exists is a runtime answer from the adapter list.
3. **Vulkan 1.3 core, no extension fallbacks for the base path.** The backend
   requires dynamic rendering (no `VkRenderPass` or `VkFramebuffer` objects),
   synchronization2 (barriers are written against the stage and access pairs the
   RHI states translate to), timeline semaphores (they are the RHI's fence), and
   descriptor indexing with runtime arrays, partial binding and variable counts
   (the bindless table of ADR 0022). A device without one is refused with the
   missing feature named. Buffer device address and indirect count are enabled
   when present and reported, because GPU-driven rendering has D3D12 fallbacks
   already. Earlier devices are an explicit non-goal: each fallback would be a
   second code path the golden-image tests would have to cover.
4. **Validation is part of the context, not an option on top.** The Khronos
   validation layer, with synchronization validation, is enabled when installed;
   every error is counted, and a test treats a non-zero count as a failure. A test
   or CI job can require the layer, so a missing one cannot pass as a clean run.
5. **Adapters are ordered like D3D12's.** Discrete GPUs, integrated GPUs, virtual
   GPUs, other, and software implementations last, so index 0 is hardware and the
   last index is what CI can rely on. The order is stable and shared by
   enumeration and device creation.
6. **Increments.** The phase is larger than one change, and the plan is recorded
   so a half-built backend cannot be mistaken for a finished one. Increment 1 (this
   ADR) is the loader, instance, physical-device choice, queue-family selection,
   feature gate, logical device, validation messenger and object naming, with
   `Backend::createDevice` answering `ErrorCode::Unsupported` until the device
   interface exists. The remaining increments are listed in the design note and
   each one lands with tests on both backends.

## Consequences

- The Vulkan backend builds with the same toolchain as everything else and adds no
  dependency to `THIRD_PARTY_NOTICES.md`.
- The cost is transcription: every new entry point or structure must be checked
  against the specification by hand. Structure layouts are the dangerous part, so
  large ones are declared as flag arrays or opaque tails, and a test creates a real
  device on every machine that has a driver, which catches a mis-sized structure
  the first time it is used.
- Adapter enumeration creates an instance per call, and a driver can take hundreds
  of milliseconds to do it (about 300 ms on the Intel HD Graphics 530 test
  machine). It happens at startup and in tests, never per frame. Caching would
  require global state, which the backend rule forbids.
- Rejecting pre-1.3 devices excludes some older Android and Linux hardware. That is
  accepted; the alternative is a render-pass and pipeline-barrier path that nothing
  on the roadmap needs.

## Alternatives considered

- **Vendor `vulkan.h` and `vk_platform.h`.** Rejected: it is a third-party
  dependency, and it would still not remove the need to load entry points
  dynamically.
- **Link `vulkan-1.lib` from the SDK.** Rejected: it makes the SDK a build
  requirement and turns a missing runtime into a startup failure.
- **Render passes and framebuffers.** Rejected: dynamic rendering maps one-to-one to
  the RHI's render pass description, and `VkRenderPass` objects would be a cache the
  RHI has no equivalent of.
- **MoltenVK for Apple platforms.** Rejected here; Phase 35 builds a Metal backend.
