# cyber_plugin_host

**Status:** active · **Layer:** 4 · **Kind:** service

Validate and load renderer plugins through the stable ABI, negotiate capabilities, and select the configured renderer with an explicit fallback diagnostic.

The binding contract for this module lives in [docs/MODULE.md](docs/MODULE.md).
Read it before adding code or changing a public header.

## Layout

- `include/cyber/plugin_host/` — public headers, self-contained.
- `src/` — private implementation.
- `tests/` — unit tests owned by this module.

The module is registered as `cyber::plugin_host`. Build the optional software
renderer plugin with `CYBER_BUILD_PLUGINS=ON`; the host itself is part of the
standard engine build.

## Roadmap

Phases 25.
