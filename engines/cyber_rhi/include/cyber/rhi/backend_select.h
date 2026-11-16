#pragma once

// Choosing a backend by name at run time, so one sample or test binary runs the
// same renderer code on D3D12 and on Vulkan (Phase 34).
//
// The order of precedence is: a `--backend=<name>` argument, then the
// `CYBER_RHI_BACKEND` environment variable, then the caller's default. Names are
// `d3d12`, `vulkan` and `null`, case-insensitive.

#include <cyber/core/types.h>
#include <cyber/rhi/backend.h>

namespace cyber::rhi
{

// The backend called `name`, or null for an unknown name. A known backend that is
// not compiled in is also null (its entry point returns null).
[[nodiscard]] const Backend* BackendByName(const char* name) noexcept;

// True for an argument this module consumes (`--backend=...`), so a program's own
// argument parser can skip it.
[[nodiscard]] bool IsBackendArgument(const char* argument) noexcept;

// The backend the command line or environment asks for, or `fallback` when neither
// names one. When a name is given but unknown, returns null and, if `error` is not
// null, points it at a static description.
[[nodiscard]] const Backend* SelectBackend(int argc, const char* const* argv,
                                           const Backend* fallback,
                                           const char** error = nullptr) noexcept;

// Copies the environment variable `name` into `out` (always terminated). Returns
// false when it is unset or does not fit.
[[nodiscard]] bool ReadEnvironmentVariable(const char* name, char* out,
                                           cyber::core::usize capacity) noexcept;

}  // namespace cyber::rhi
