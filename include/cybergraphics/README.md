# cybergraphics — stable product API

This directory is the **only** public SDK surface. It contains the versioned C
ABI that Cyber Engine uses to load a renderer plugin.

Rules:

- C types only. No `std::`, templates, exceptions, RTTI, or C++ classes.
- Every struct starts with `size` and `version`.
- Objects are opaque handles.
- A plugin exports one versioned entry point returning a function table.
- Recoverable errors are `cg_result` codes.
- Capability negotiation precedes optional features.
- No implementation lives here. The GPU and software renderers implement this
  contract.

The public declarations are in [`api.h`](api.h). The full policy is in
[`docs/architecture/abi-policy.md`](../../docs/architecture/abi-policy.md),
and the plugin loading and compatibility rules are in
[`docs/plugin-abi.md`](../../docs/plugin-abi.md).

Phase 25 is complete: the software and D3D12 plugins, host-side selection and
fallback, and runtime compatibility tests are implemented. Phase 26 is now in
progress; its PBR renderer design and implementation status are documented in
[`docs/graphics/PHASE_26.md`](../../docs/graphics/PHASE_26.md).
