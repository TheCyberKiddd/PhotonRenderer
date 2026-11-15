# Photon Renderer

Photon Renderer is an experimental C++20 real-time rendering project. It
combines a deterministic CPU software renderer, a Direct3D 12 rendering
interface and backend, scene and asset utilities, and runnable feature samples.

The product is named **Photon Renderer**. Existing implementation identifiers
such as CMake targets, modules, and sample executables still use the
`cyber_*` naming scheme.

## Project status

The 44-phase roadmap is complete through **Phase 33: Texture compression and
streaming**. **Phase 34: Vulkan backend** is complete for the RHI (renderer code runs on Vulkan as on D3D12); SPIR-V output from the shader compiler is the remaining piece. This is a pre-release
project, not a production-ready renderer; feature coverage varies by backend.

Current highlights:

- CPU reference rendering with rasterization, depth, texturing, physically
  based shading, image-based lighting, shadows, HDR post-processing, temporal
  anti-aliasing, and baked global illumination.
- A Direct3D 12 backend with GPU compute culling, LOD selection, Hi-Z occlusion,
  and indirect drawing.
- BC1, BC4, BC5, and BC7 texture codecs, budgeted asynchronous mip streaming,
  and experimental CPU virtual-texture support.
- Asset import and cooking, renderer plugins, and samples ranging from basic
  triangles to a million-instance GPU-driven scene.

Important limitations:

- Direct3D 12 is the only implemented GPU backend. Vulkan and Metal are planned,
  not available.
- The Phase 29-31 visual-quality features are implemented for the CPU
  reference-renderer scope; they do not imply matching GPU implementations.
- Phase 33 compression, streaming, and virtual-texture features currently use
  CPU-side residency and feedback. The page pool is not bound to D3D12 reserved
  resources, and hardware sampler-feedback streaming is not implemented.
- Ray tracing currently means CPU reference tracing and baking. GPU ray-tracing
  acceleration-structure builds and dispatch are not implemented.
- Meshlets have a vertex-pipeline fallback; a mesh-shader pipeline is not
  implemented.

See the [roadmap status](docs/ROADMAP_STATUS.md), [roadmap index](docs/ROADMAP.md),
and [Phase 33 report](docs/reports/phase-33.md) for scope and evidence. Formal
packaging, license review, beta support, and release planning remain future
work (roadmap Phases 43-44).

## Build and test

### Windows

Install Visual Studio 2022 with the Desktop C++ workload, CMake 3.28 or newer,
and Ninja. From a PowerShell prompt at the repository root:

```powershell
cmake --preset ci-windows
cmake --build --preset ci-windows
ctest --preset ci-windows --output-on-failure
```

For a Release build:

```powershell
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
```

### Linux

The CI Linux presets use Clang, Ninja, and UBSan or ASan:

```sh
cmake --preset ci-linux
cmake --build --preset ci-linux
ctest --preset ci-linux --output-on-failure
```

## Run a sample

The showcase renders the software-reference feature set to PPM images:

```powershell
cmake --preset release
cmake --build --preset release --target cyber_sample_showcase
.\build\release\samples\13_showcase\cyber_sample_showcase.exe
```

The GPU-driven sample requires a Direct3D 12 adapter:

```powershell
cmake --build --preset release --target cyber_sample_gpu_driven
.\build\release\samples\17_gpu_driven\cyber_sample_gpu_driven.exe
```

See [all samples](samples/README.md) for descriptions and
[showcase instructions](samples/13_showcase/README.md). Phase 33 also provides
a [CPU virtual-texture sample](samples/19_virtual_texture/README.md).

## Continuous integration

GitHub Actions checks formatting and module boundaries, builds and tests with
MSVC on Windows, and builds and tests with Clang plus sanitizers on Linux. It
also verifies installed-package and source-tree consumers. Workflow details
are in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## License

Photon Renderer is distributed under the project-specific terms in
[LICENSE](LICENSE).
