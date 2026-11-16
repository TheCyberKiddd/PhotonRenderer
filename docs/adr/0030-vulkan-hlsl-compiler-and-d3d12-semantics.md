# 0030 — Vulkan compiles HLSL itself, and keeps D3D12's defined behaviour

- **Status:** accepted
- **Date:** 2026-11-16
- **Deciders:** @TheCyberKiddd
- **Related:** Phase 34, [design note](../design/phase-34-vulkan.md), ADR 0019
  (shader strategy), ADR 0029 (Vulkan backend)

## Context

Phase 34's done line is "every sample and golden-image test passes on both D3D12 and
Vulkan using the same renderer code". The renderer's shaders are HLSL: written by hand
in the renderer and samples, or generated from the Cyber shader language (ADR 0019).
The D3D12 backend accepts `ShaderBytecodeFormat::HlslSource` and compiles it with the
compiler Windows ships. Vulkan consumes SPIR-V. The Windows SDK's `dxc.exe` on the
development machine was built without SPIR-V code generation, and the project avoids
third-party toolchains (ADR 0019).

Running the same renderer code also showed that D3D12 *defines* things Vulkan leaves
undefined, and that renderer code relies on them: an out-of-range shader read, a
float-to-integer conversion out of range, the contents of a newly created resource.
On the HD Graphics 530 each undefined case was a wrong result or a device lost.

## Decision

1. **The Vulkan backend accepts HLSL source and compiles it in the project**
   (`src/vulkan/hlsl_spirv*`). It is a compiler for the HLSL subset the engine uses
   (structs, cbuffers and `ConstantBuffer<T>`, structured and byte-address buffers,
   textures and samplers including arrays and `NonUniformResourceIndex`, storage
   images, groupshared memory, atomics, barriers, the math intrinsics, control flow,
   functions with in/out parameters). Anything outside the subset fails with a
   located diagnostic that says it is not supported; it never emits a guess. It lives
   in `cyber_rhi` because `cyber_shaders` sits above the RHI in the module order, and
   because it is the same contract as D3D12's `HlslSource`: the backend compiles what
   it is given. The Cyber language therefore reaches SPIR-V as Cyber → HLSL → SPIR-V,
   all in project code.
2. **Shaders are compiled per pipeline, against its layout.** HLSL names registers;
   Vulkan names sets and bindings. The device keeps the HLSL and compiles it when a
   pipeline is created, resolving `register(<class><n>, space<s>)` through the
   pipeline layout with the register assignment the D3D12 root signature uses, and
   `b0, space0` as the push-constant block when the layout has push constants. Vertex
   outputs and pixel inputs are linked by semantic, not by declaration order. Source
   is still compiled once at `CreateShader`, so syntax and type errors surface there as
   on D3D12.
3. **The generated code keeps HLSL's and D3D12's meaning**: HLSL matrix orientation
   and column-major packing, cbuffer and structured-buffer packing as explicit
   offsets, saturating float-to-integer conversion with NaN to 0, `SV_Position.w` as
   clip w in pixel shaders, `SV_InstanceID` from 0, and scalar `&&`, `||` and `?:`
   that evaluate only the operand they need (as DXC does), so a guard protects the
   read it guards.
4. **The device keeps D3D12's defined behaviour**: `robustBufferAccess` and
   `robustImageAccess` are enabled (out-of-range reads return zero, writes are
   dropped); new textures and buffers are zero-filled, as committed D3D12 resources
   are; binding arrays are update-after-bind and partially bound, so they reach the
   descriptor-indexing limits as D3D12 descriptor tables do.

## Consequences

### Positive

- Every renderer GPU test, every RHI sample and the golden-image comparison run on
  D3D12 and Vulkan from one binary each, selected by `--backend=` or
  `CYBER_RHI_BACKEND`; CTest registers both.
- No new dependency. The compiler is fuzzed in `rhi_hlsl_spirv` and runs on every CI
  machine, GPU or not.
- Zero-filled creation removed an intermittent GPU hang: on Intel, an image placed in
  reused memory inherited stale compression metadata.

### Negative

- A second HLSL implementation to keep in step with what the renderer writes. The
  subset grows only when a shader needs it.
- Pipelines compile HLSL at creation, a cost D3D12 pays at `CreateShader`. Pipeline
  caches (now implemented) keep the driver half of that cost across runs.
- Robust access and zero-filled creation cost a little bandwidth and a clear per
  resource; both are what D3D12 already pays.

## Alternatives considered

| Alternative | Reason not selected |
|---|---|
| A SPIR-V-enabled DXC build | A download and a third-party toolchain (ADR 0019); the machine's DXC has no SPIR-V. It remains a drop-in: precompiled SPIR-V is still accepted. |
| SPIR-V generation from the Cyber language's IR only | The renderer's hand-written HLSL would not run on Vulkan, so the done line could not be met. |
| Hand-written SPIR-V per shader | The approach the parity test used before this ADR; not "the same renderer code". |
| Leaving Vulkan's undefined behaviour to the renderer | Renderer code would differ per backend, and the failures found were hangs, not errors. |
