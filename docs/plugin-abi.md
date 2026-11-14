# Renderer plugin ABI

The public C header, [`include/cybergraphics/api.h`](../include/cybergraphics/api.h),
is the source of truth for the renderer plugin boundary. The ABI is intentionally
separate from the internal C++ module APIs.

## Loading and negotiation

Each renderer library exports exactly one entry point,
`cgGetPluginApi`. The host resolves it as `cg_get_plugin_api_fn` and supplies
the ABI version it supports and a zero-initialized `cg_plugin_api` whose `size`
is set to the number of bytes the host allocated and whose `version` is set to
the structure version it understands.

A plugin must reject an unsupported ABI major version with
`CG_ERROR_INCOMPATIBLE_VERSION`. On success, it fills only the table prefix
that fits in the caller's `size`; the returned `size` is the number of bytes
initialized and must never exceed the incoming capacity. It reports its table
version and leaves unknown appended entries untouched. The host must verify
that the returned size reaches every function it plans to call, then query
`cg_plugin_info` before creating resources. A missing optional capability or
function is not a successful substitute for the requested operation.

The single exported entry point has this declaration:

```c
CG_PLUGIN_EXPORT cg_result CG_CALL cgGetPluginApi(
    cg_uint32 requested_abi_version, cg_plugin_api* in_out_api);
```

`CG_PLUGIN_BUILD` enables the platform export annotation when compiling a
plugin. The host must keep the dynamic library loaded until after the device
and every object created by that plugin have been destroyed.

## Data, ownership, and lifetime

- ABI scalar types have explicit widths. Structures begin with `size` and
  `version`; callers initialize both before every call.
- Matrices are 16 column-major `float` values, matching CyberGraphics' math
  convention. Vectors and colors are tightly packed float arrays.
- Input arrays and strings are borrowed for the duration of the call only.
  Plugins copy data they need to retain. The host owns screenshot storage and
  supplies its capacity and row stride.
- Plugin-info names point to immutable plugin-owned storage and remain valid
  only while the library is loaded. Screenshot format and stride describe the
  caller's buffer; `required_bytes` reports the minimum capacity for the
  requested dimensions and format.
- Handles are opaque and belong to the plugin that created them. Do not mix
  handles between plugins or destroy a resource while a live object references
  it. Destroy child views, instances, and lights before their scene; destroy
  scenes and resources before the device.
- No C++ allocation, exception, RTTI, STL type, or ownership crosses the ABI.
  Plugins release all plugin-owned memory through the same module that
  allocated it.
- Recoverable errors are `cg_result` values. If an error callback was supplied
  at device creation, the plugin may provide a diagnostic message; the message
  is borrowed and valid only for the callback. Callbacks are synchronous on
  the calling thread and must not be used to call back into the plugin.
- A successful function must fully initialize its documented outputs. Failed
  calls leave output handles null and output sizes set when required to report
  capacity.

## Compatibility rules

ABI version 1 fixes the calling convention, result codes, handle semantics, and
the existing function-table order. Never reorder or change the meaning of an
existing entry. Add optional entries only at the end of the table and gate
their use by the returned table size and capability bits. An incompatible
calling convention or changed existing entry requires a new ABI major version.

Structure versions identify layout generations. Additive fields are appended;
read a field only when `size` covers that field. Unknown trailing bytes are
ignored. ABI version 1 operations require the fields they read. The software renderer
accepts the original `cg_device_desc` prefix ending before `native_window`;
presentation-capable plugins require the appended field and must reject a
missing or null native surface. Future structure extensions must preserve the
version-1 prefix and explicitly support shorter prefixes before claiming that
compatibility.

The host selects a renderer from an explicit startup preference, validates the
entry point, ABI, required function-table prefix, and requested capabilities,
then creates the device. `RendererInstance::Create` tries the configured
preference first and falls back to the other plugin after load, negotiation, or
device-creation failure. Partial device state is destroyed before fallback;
both preferred and fallback results remain available in
`RendererLoadDiagnostics`.

## Current implementation boundary

The implementation includes the public contract and C11 consumer test,
loadable software and D3D12 plugin products, host-side loading and capability
negotiation, and startup selection with fallback after plugin loading,
negotiation, or device creation. The D3D12 plugin presents through the borrowed
native window, submits scene geometry through the RHI, and supports resizing.
It does not advertise lighting, debug-line, or screenshot capabilities.

Runtime tests cover legacy function-table negotiation, the original software
device-descriptor prefix, software rendering and screenshots, preferred-plugin
failure and fallback, D3D12 error reporting, and a hidden-window D3D12 triangle
submission. The plugin host exposes renderer preference as a startup API; this
repository has no separate engine executable or persisted settings UI to wire.
