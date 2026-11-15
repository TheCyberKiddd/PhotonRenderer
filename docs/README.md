# CyberGraphics documentation

This directory is the project's memory. Code is only half of the design; the
other half is the written contract, the decision record, and the evidence that
a phase actually works.

## Start here

| Document | Purpose |
|---|---|
| [ROADMAP.md](ROADMAP.md) | Master checklist for all 44 phases, linked to the two binding plans. |
| [requirements.md](requirements.md) | Project-level functional and non-functional requirements. |
| [architecture/overview.md](architecture/overview.md) | The module map, layers, and dependency law. |
| [architecture/integration.md](architecture/integration.md) | How Cyber Engine consumes CyberGraphics as a submodule or package. |
| [architecture/layers.md](architecture/layers.md) | The exact layer and order rules and how CI checks them. |
| [architecture/module-contract.md](architecture/module-contract.md) | Template and checklist for every module. |
| [architecture/threading.md](architecture/threading.md) | Thread ownership and frame model. |
| [architecture/error-model.md](architecture/error-model.md) | `Result`, assertions, error codes, no exceptions. |
| [architecture/abi-policy.md](architecture/abi-policy.md) | The stable plugin boundary. |
| [math-conventions.md](math-conventions.md) | Handedness, matrices, depth, units, color. |
| [coding-standard.md](coding-standard.md) | Naming, headers, ownership, style. |
| [glossary.md](glossary.md) | Terms used in code and docs. |
| [debugging.md](debugging.md) | D3D12 validation, object labels, and PIX/RenderDoc capture notes. |

## Working documents

| Directory | Contents |
|---|---|
| [graphics/](graphics/) | The binding CyberGraphics roadmap and per-phase build guide. |
| [design/](design/) | One design note per roadmap phase, written before code. |
| [adr/](adr/) | Architecture decision records. |
| [notes/](notes/) | Reading notes and sources for each phase. |
| [reports/](reports/) | Phase reports with evidence for the done-when line. |
| [graphics/PHASE_17.md](graphics/PHASE_17.md) | Phase 17 implementation summary and validation status. |
| [graphics/PHASE_18.md](graphics/PHASE_18.md) | Phase 18 material-system implementation summary and validation status. |
| [reports/phase-18.md](reports/phase-18.md) | Phase 18 report and local validation evidence. |
| [graphics/PHASE_20.md](graphics/PHASE_20.md) | Phase 20 render-graph implementation summary and validation status. |
| [reports/phase-20.md](reports/phase-20.md) | Phase 20 report and local validation evidence. |
| [graphics/PHASE_21.md](graphics/PHASE_21.md) | Phase 21 render-thread implementation summary and validation status. |
| [graphics/PHASE_24.md](graphics/PHASE_24.md) | Phase 24 asset-pipeline implementation summary and validation status. |
| [reports/phase-24.md](reports/phase-24.md) | Phase 24 report and local validation evidence. |
| [reports/phase-21.md](reports/phase-21.md) | Phase 21 report and local validation evidence. |
| [design/phase-29-ibl-reflections.md](design/phase-29-ibl-reflections.md) | Phase 29 IBL, reflection ownership, algorithms, and software-reference scope. |
| [notes/phase-29.md](notes/phase-29.md) | Phase 29 reading notes and source links. |
| [reports/phase-29.md](reports/phase-29.md) | Phase 29 implementation and validation evidence. |
| [design/phase-30-post-processing.md](design/phase-30-post-processing.md) | Phase 30 post chain, anti-aliasing paths, and how the ghosting bar is measured. |
| [reports/phase-30.md](reports/phase-30.md) | Phase 30 implementation and validation evidence. |
| [adr/0028-gi-approach.md](adr/0028-gi-approach.md) | Phase 31 choice of baked irradiance volumes, and the ray-tracing capability gate. |
| [adr/0029-vulkan-backend.md](adr/0029-vulkan-backend.md) | Phase 34 choice of own Vulkan declarations loaded at run time, the 1.3-core requirement, and the increment plan. |
| [design/phase-34-vulkan.md](design/phase-34-vulkan.md) | Phase 34 status, what remains, and D3D12/Vulkan behaviour differences. |
| [design/phase-31-gi-raytracing.md](design/phase-31-gi-raytracing.md) | Phase 31 ray tracer, irradiance volume, denoiser, and how GI accuracy and leakage are measured. |
| [reports/phase-31.md](reports/phase-31.md) | Phase 31 implementation and validation evidence. |
| [design/phase-32-gpu-driven.md](design/phase-32-gpu-driven.md) | Phase 32 GPU culling, Hi-Z occlusion, indirect-count draws, meshlets, async compute, and how the GPU is held to the CPU culler. |
| [reports/phase-32.md](reports/phase-32.md) | Phase 32 implementation and validation evidence. |
| [design/phase-33-compression-streaming.md](design/phase-33-compression-streaming.md) | Phase 33 BCn codec, PSNR floors, the page-pool residency manager, eviction without thrashing, and how hitches are measured. |
| [reports/phase-33.md](reports/phase-33.md) | Phase 33 implementation and validation evidence, and the software renderer performance pass. |
| [graphics/PHASE_15.md](graphics/PHASE_15.md) | Phase 15 implementation summary and capture status. |
| [specs/](specs/) | RHI, shader language, and asset format specifications. |
| [testing/](testing/) | Test layers, golden images, sanitizers, and CI gates. |
| [perf/](perf/) | Performance log and budgets. |

## Rules for documentation

- Write design notes before implementation, not after.
- List sources: specifications, papers, talks. Ideas are allowed; copied code
  is not.
- Record decisions as ADRs when they are expensive to reverse.
- Phase reports must name the machine and include the evidence, not a promise.
- Keep the glossary current; inconsistent vocabulary becomes inconsistent code.
