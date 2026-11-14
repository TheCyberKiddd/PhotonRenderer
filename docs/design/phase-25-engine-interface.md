# Phase 25: engine-facing renderer interface

Status: complete. The versioned C ABI, software and D3D12 plugins, host
selection and device-creation fallback, and runtime/package validation are
implemented. Renderer choice is an explicit host startup preference; this
repository does not contain a separate engine executable or persisted settings
UI.

## Boundary

Cyber Engine describes a frame using the stable C API in
[`include/cybergraphics/api.h`](../../include/cybergraphics/api.h). The table
covers device lifetime and resize, scene/view setup, copied mesh data,
materials, instances, lights, debug lines, frame submission, and
caller-owned screenshot output. A host talks to a renderer only through this
table; it does not depend on internal renderer C++ headers.

The API defines a renderer-neutral contract rather than exposing the RHI or
software rasterizer directly. Backend-specific facilities must be advertised
as capabilities. The initial contract keeps scene content CPU-readable so the
software renderer can be a genuine independent implementation.

## Decisions

1. Use a C ABI with one exported `cgGetPluginApi` entry point returning an
   append-only function table. This keeps symbol lookup small and permits
   plugins to add optional operations without changing existing offsets.
2. Use fixed-width integer results and flags, pointer-sized opaque handles,
   caller-owned output buffers, and plugin-owned resource allocations. C++
   types and allocator ownership do not cross the boundary.
3. Begin with ABI major version 1. Every structure carries `size` and
   `version`; table size and capability bits gate optional features. A major
   mismatch is rejected before device creation.
4. Make input pointers borrowed for the call and require retained input to be
   copied by the plugin. Screenshot memory is host-owned and sized by an
   explicit capacity and stride.
5. Keep renderer preference and fallback in the host. A fallback is permitted
   only after a clear plugin-load, compatibility, capability, or device-create
   failure, and the original diagnostic must remain visible.

## Validation plan

- Compile and execute a C11 consumer that includes the public header.
- Test export lookup, ABI mismatch, undersized function tables, capability
  negotiation, and host/plugin builds using compatible function-table and
  device-descriptor sizes.
- Exercise software rendering, invalid arguments, screenshot buffer sizing,
  resize, D3D12 error callbacks, and hidden-window indexed triangle submission.
- Exercise explicit renderer selection and fallback after device-creation
  failure in the host startup path.
- Run package and source consumers to ensure the public header and target are
  available in both integration modes.

## Implementation and validation

The host validates the ABI, table prefix, requested capabilities, and plugin
info before device creation. `RendererInstance::Create` destroys partial
preferred-device state and attempts the alternate plugin when the preferred
plugin cannot load, negotiate, or create a device. Device-creation results are
retained in `RendererLoadDiagnostics`.

The software plugin implements scene/view/mesh/material/instance creation,
frame submission, resize, and screenshots. The Windows D3D12 plugin uses the
borrowed `native_window` HWND, submits indexed scene geometry through the RHI,
and supports resize. It does not advertise lighting, debug-line, or screenshot
support. Tests verify software output and descriptor-prefix compatibility,
legacy function-table negotiation, fallback from an unsupported D3D12 surface
to software, callback reporting, and a real hidden-window D3D12 triangle
submission. The D3D12 integration test skips only when no D3D12 adapter is
available.

There is no standalone engine executable in this repository; the host startup
API accepts `RendererPreference` so an integrating engine can provide its
configuration value directly.

## References

- [ABI policy](../architecture/abi-policy.md)
- [Integration contract](../architecture/integration.md)
- [ADR 0003: integration and packaging](../adr/0003-integration-and-packaging.md)
- [ADR 0013: RHI object model](../adr/0013-rhi-object-model.md)
- [ISO/IEC 9899:2018, Programming Languages — C](https://www.iso.org/standard/74528.html)
- [Microsoft Learn: LoadLibraryW](https://learn.microsoft.com/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryw)
- [Microsoft Learn: GetProcAddress](https://learn.microsoft.com/windows/win32/api/libloaderapi/nf-libloaderapi-getprocaddress)
- [`dlopen(3)`](https://man7.org/linux/man-pages/man3/dlopen.3.html) and
  [`dlsym(3)`](https://man7.org/linux/man-pages/man3/dlsym.3.html)
