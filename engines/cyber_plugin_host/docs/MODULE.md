# cyber_plugin_host — module contract

**Kind:** service
**Layer:** 4
**Order:** 42
**Status:** active
**Owner:** @TheCyberKiddd

## Purpose

Validate and load renderer plugins through the stable ABI, negotiate capabilities, and select the configured renderer with an explicit fallback diagnostic.

## Public API

`include/cyber/plugin_host/`

Only headers under this path are public. Everything under `src/` is private.
Public headers must compile on their own and must not include another module's
private headers.

## Dependencies

Allowed: cyber_core, cyber_platform, cybergraphics_api

Forbidden: cyber_software_renderer, cyber_rhi

Dependencies are one-way. A module may depend only on modules with a lower
`order` value, or on a declared contract such as `cybergraphics_api`.

## Invariants

- Keep plugin libraries loaded for the entire lifetime of plugin-owned handles.
- Reject incompatible ABI tables and unmet required capabilities before device creation.
- Preserve the preferred renderer failure when a fallback is selected.

## Non-goals

- Renderer implementation.
- Automatic fallback after device creation has begun.

## Tests

Dynamic plugin ABI negotiation, capability validation, renderer preference, fallback diagnostics, and plugin lifetime.

## Budgets

No per-frame allocation or plugin loading after startup selection.

## Roadmap

Phases 25.
