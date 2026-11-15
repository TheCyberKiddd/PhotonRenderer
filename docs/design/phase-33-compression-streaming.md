# Phase 33 design — Texture compression and streaming

Date: 2026-11-15
Status: Complete.
Decoders for BC1 to BC7 (BC6H included) define the reference output, and the
BC6H decoder is checked bit for bit against the hardware decoder; encoders are
held to PSNR floors in CI. The residency manager streams mips
into a fixed page budget on background threads and is exercised by a sample
with 16.7 times more texture data than its budget. The RHI has a
reserved-texture (tiled resource) API; `GpuTexturePool` backs the streamer with
a real tile heap on D3D12, and a compute shader samples the streamed textures
with every texel verified against the source. `VirtualTexture` is mirrored to
the GPU by `GpuVirtualTexture`, whose HLSL sampler matches the CPU lookup
exactly on both test adapters, and `GpuVirtualTextureFeedback` closes the
loop by having that shader report the pages it needs.
Related: [Phase 33 roadmap](../graphics/CYBERGRAPHICS_ROADMAP.md#phase-33-texture-compression-and-streaming),
[Phase 33 report](../reports/phase-33.md),
[Phase 9 design](phase-09-textures-shaders.md),
[Phase 11 design](phase-11-resources.md)

## Scope and boundaries

The roadmap asks for our own BCn compressor, mip streaming, a residency manager
with a memory budget, and background loading threads, and is done when a scene
with more texture data than the video memory budget runs without hitches and a
residency view shows what is loaded. The build guide adds BCn decoders first,
PSNR measured against the decoder, a texture pool allocator, an eviction
policy, low-resolution placeholders while data loads, and PSNR thresholds in
tests.

In scope:

- `cyber/assets/bc_codec.h`: block and surface decoders and encoders for BC1
  to BC5, BC6H and BC7, mip-chain generation (linear or sRGB-correct), and PSNR.
- `cyber/renderer/texture_streaming.h`: `TextureStreamer` (the residency
  manager, page pool, IO threads, eviction), `ComputeWantedMip`, and
  `RenderResidencyView`.
- `cyber/platform/thread.h`: `Thread::SetCurrentThreadPriority`, so IO threads
  run below the frame.
- `samples/18_texture_streaming`: an oversubscribed corridor flythrough with the
  hitch, budget, and lag measurements, and residency-view images.

Out of scope, and stated rather than implied:

- **Hardware tiled virtual texturing.** The virtual texture resolves pages in a
  shader through a page-table texture and an atlas, the software scheme engines
  use when they need to run on every adapter. It does not map hardware tiles for
  the virtual image. The page *feedback* is hardware independent too: see
  "GPU virtual texture feedback" for why D3D12 sampler feedback is not used.

## The codec

### Decoders define the output

The decoders were written first and are the reference everything else is
scored against. They follow the Direct3D 11 functional specification:

- BC1 expands 565 endpoints by bit replication and interpolates in integers
  with round-to-nearest: `(2a + b + 1) / 3`, and `(a + b + 1) / 2` in
  three-colour mode, whose fourth entry is transparent black.
- BC4 uses `((8 - k) a + (k - 1) b + 3) / 7` in eight-value mode and
  `/ 5` with exact 0 and 255 in six-value mode. BC5 is two BC4 blocks.
- BC7 implements all eight modes, partition tables, anchor indices, p-bits,
  rotation and index selection, and interpolates with the specification's 6-bit
  weights `((64 - w) e0 + w e1 + 32) >> 6`. Unlike BC1 and BC4, BC7 leaves no
  rounding freedom, so this decoder is bit-exact with any conforming one.
  Reserved mode bytes decode to transparent black.

The partition tables were checked against the specification's anchor tables:
every listed anchor is where the table's second (and third) subset is first
indexed with the reduced bit count, and the BC7 round trips would fail their
PSNR floors if any entry were wrong.

### Encoders

Every encoder scores candidates with the decoder's exact palette, so the error
it minimizes is the error the GPU will show.

- **BC1**: endpoints from a principal-component line fit of the opaque texels,
  then alternating index assignment and least-squares endpoint refits.
  Four-colour and three-colour modes are both searched (three-colour is
  mandatory when a texel is below the punch-through threshold). Solid blocks
  use optimal single-colour tables: the 5- and 6-bit endpoint pairs whose
  two-thirds interpolant reproduces each 8-bit value, which keeps a flat colour
  within the 565 interpolation grid instead of losing up to four codes.
- **BC4**: both modes are searched; the six-value mode fits its endpoints to
  the values strictly between 0 and 255, which it then stores exactly.
- **BC2 and BC3**: 16-byte blocks of alpha then colour. BC2 rounds alpha to 4
  bits; BC3 stores alpha as a BC4 block. The colour block is always decoded in
  four-colour mode whatever the endpoint order, so the encoder takes the BC1
  encoder's endpoints and re-picks every index against that four-colour
  palette, never trusting an index the BC1 encoder chose for three-colour mode.
- **BC7**: opaque blocks try mode 6, then modes 1 and 3 on the partitions a
  cheap estimate ranks best, and modes 0 and 2 on three-subset partitions at
  High (mode 2 over all 64 partitions). Blocks with alpha try modes 6 and 5
  (all four channel rotations of mode 5 at High),
  then mode 4 (colour 5 bits and alpha 6 bits with separate 2- and 3-bit index
  sets, both index selections, all four channel rotations at High) and mode 7
  (two RGBA subsets) above Fast. The partition estimate is the summed
  line-fit residual of each subset — total variance minus variance along the
  principal axis — which costs a 3x3 power iteration per subset instead of a
  full encode. Solid blocks use a mode 5 single-colour table (7-bit endpoints
  at weight 21/64 reach every 8-bit value), so flat colour is exact; mode 6
  cannot promise that because its p-bit is shared by an endpoint's channels.
  P-bits are chosen per endpoint or per pair by quantization error, and
  exhaustively at High.
- **BC6H**: see "BC6H" below; it works on half-float RGB, not RGBA8.

`BcQuality` sets the refinement passes and, for BC7, how many partitions are
tried: Fast 1, Normal 4, High 12 plus four three-subset ones (modes 4 and 7
and the wide mode 2 search are skipped at Fast, so it stays the fast path). Quality never
decreases with the level; a test holds that.

### Holding quality in CI

`assets_bc_codec` encodes a 128x128 synthetic test card — smooth gradients,
hard two-colour stripes, a hue ring, and textured noise — and checks PSNR
against the decoder. The floors sit about half a decibel under the measured
values, so a change that lowers quality fails:

| Format | Quality | Measured | Floor |
|---|---|---|---|
| BC1 (RGB) | Normal | 37.44 dB | 37.0 dB |
| BC4 (R) | Normal | 51.72 dB | 51.0 dB |
| BC5 (RG) | Normal | 51.68 dB | 51.0 dB |
| BC7 (RGBA, opaque) | Fast | 43.48 dB | 43.0 dB |
| BC7 (RGBA, opaque) | Normal | 43.53 dB | 43.0 dB |
| BC7 (RGBA, opaque) | High | 43.62 dB | at least Normal |
| BC7 (RGBA, alpha ramp) | Fast | 40.08 dB | none |
| BC7 (RGBA, alpha ramp) | Normal | 41.34 dB | 40.8 dB |
| BC7 (RGBA, alpha ramp) | High | 42.03 dB | 41.5 dB |
| BC2 (RGBA, alpha ramp) | Normal | 36.45 dB | 36.0 dB |
| BC3 (RGBA, alpha ramp) | Normal | 38.68 dB | 38.0 dB |
| BC3 (RGB, opaque) | Normal | 37.46 dB | within 1 dB of the BC1 floor |

Encoding is deterministic — the tests compress twice and compare bytes — so
cooked assets are reproducible.

### Mips

`GenerateMipChain` builds a full chain with a 2x2 box filter. An odd dimension
folds its last row or column into the final texel (a 3-tap box there) so no
texel is dropped. sRGB textures are averaged in linear light: a black and white
checker filters to sRGB 188, not 128.

## Streaming

### The page model

Memory is a pool of 64 KiB pages — the D3D12 tile size — sized by the budget. A
level occupies whole pages, which need not be contiguous, so the pool cannot
fragment and the page list is exactly what `UpdateTileMappings` takes.

The coarsest levels, each smaller than a page, form the **packed mip tail**,
the same split D3D12 makes for reserved resources. The tail is requested at
registration, loaded as one request into its own page or two, and never
evicted. It is the low-resolution placeholder: every registered texture can be
drawn as soon as its tail lands, and a level load (`Flush` after registering)
guarantees that before the first frame.

Finer levels stream one at a time per texture, coarsest first. That keeps each
texture's resident levels a contiguous chain from the tail, so the sampler
clamp is one number — `ResidentLevel`, used as the SRV's `MinLOD` — and
streaming the smallest mips first means the picture sharpens progressively
instead of waiting for the largest level.

### What each texture needs

The renderer calls `ReportUsage(texture, wantedMip, coverage)` for every use in
a frame. `ComputeWantedMip` is log2 of texels per screen pixel along the
surface's larger axis — the mip the hardware would sample — with `uvScale` for
tiling. Multiple uses of a texture take the finest mip and sum their coverage.
A texture not reported for `unusedFrameLimit` frames wants only its tail.

### Scheduling

Each `Update`:

1. **Drain**: finished reads are collected from the IO threads and applied in
   priority order until `maxUploadBytesPerFrame` is spent (at least one per
   frame, so a level larger than the budget still progresses). Applying calls
   the sink's `UploadLevels` with the data and its pages.
2. **Wanted levels** are refreshed from the frame's reports, with `mipBias`.
3. **Schedule**: tails first, then every texture whose resident level is
   coarser than its wanted level, in priority order, each asking for the next
   finer level. Priority is importance (`1 + 1000 * coverage`) times one plus
   the number of missing levels, so a texture far from what it needs outranks
   one a level short.

Reads run on `ioThreads` background threads at below-normal priority. Pages and
staging memory are taken on the main thread when the request is issued, so the
IO threads never allocate; staging blocks are recycled in power-of-two classes,
because allocating and freeing a 2 MiB block per top-mip load put a
`VirtualAlloc`/`VirtualFree` pair and fresh page faults on the main thread.

### Eviction

When a request does not fit, the streamer plans evictions from a victim list
ordered by surplus first (textures holding levels finer than they now need),
then by lowest importance:

- surplus levels are always eligible;
- a needed level goes only to a request whose importance exceeds its owner's by
  25% (plus a small constant), so near-equal textures settle instead of
  trading memory;
- a texture never evicts its own chain, and a texture with a read in flight is
  skipped so its chain stays contiguous;
- the tail is never evicted.

The plan is committed only if it frees enough pages for the request. The first
version evicted one level at a time and stopped when it ran out of victims;
when the victims together could not free enough, it had evicted for nothing,
and the victims loaded the same levels back the next frame. In the unit test
that shape produced 112 evictions where the settled answer needs a handful;
the test now fails on more than eight.

Comparing importance rather than priority is deliberate. Priority grows with
the deficit, so a texture that loses a level would immediately outrank the one
that took it, and two equally important textures would evict each other
forever.

### Hitches

The streamer's main-thread work per frame is linear in the texture count plus
at most `maxUploadBytesPerFrame` of copying into the sink. Both are bounded by
configuration, which is what makes "without hitches" a property of the design
rather than of the scene. The sample sets the upload budget to one top mip
(2 MiB of BC1) per frame.

### The residency view

`RenderResidencyView` draws a budget bar (resident pages, pages reserved for
reads in flight, free) and one cell per texture: a box per streamed level, left
to right from level 0, plus one box for the whole packed tail, coloured by
state — green resident and needed, blue resident but finer than needed, yellow
loading, red needed and missing, grey not needed, teal the tail — with a bar
under the strip showing the resident share of the full chain. The sample writes
it as images; a renderer overlays it.

## Determinism and threading

The codec is single-threaded and deterministic; its block functions are
reentrant, so a cooker can split a surface across threads by block rows.

The streamer's public functions are main-thread only. The IO threads touch only
the request they popped and the source's `ReadLevel`; the queue and the
completed list are guarded by one mutex. `Unregister` waits for the texture's
read in flight before returning, so a source can be destroyed straight after.
Which read finishes first can vary between runs; what is resident after the
same reports and a `Flush` does not.

## Known limitations

- Tiling differs by driver: the HD Graphics 530 packs small mips together while
  WARP gives each its own tile. `TextureTiling` reports it and the pool handles
  both, but only these two adapters have been tested.
- On WARP, a *constant* index into a bound array of texture views (`Tex[3]`)
  returns element 0. See "WARP and constant array indices" below: it is the
  adapter, not the RHI or a compiler, and run-time indices (how bindless access
  is written) are correct.
- Creating a device with `Exhaustive` validation after another device existed in
  the same process failed on one adapter in `rhi_d3d12_reserved`, which uses
  `Standard` for that reason. GPU-based validation is process-wide and set before
  device creation, which is the likely cause; it is not isolated.
- BC6H encodes with a squared-error search in the half-float integer domain
  and has no perceptual weighting. It is not tuned against another encoder.
- BC7 mode 5 and mode 4 rotations are searched only at High.
- Streaming moves one level per texture at a time. That keeps chains
  contiguous and the logic simple; a texture that needs several levels at once
  fills them over consecutive reads, not in parallel.
- The streamer's own wanted mips still come from analytic screen coverage;
  `GpuVirtualTextureFeedback` feeds the virtual texture only. A
  `TextureStreamer` fed from a shader-written feedback buffer is the same
  technique applied to `ReportUsage`.

## Reserved textures and the GPU pool

**RHI.** `Feature::ReservedTextures` gates a tiled-resource API:
`TextureDesc::reserved` creates a 2D texture with no memory behind it;
`CreateTileHeap` makes a heap of 64 KiB tiles (`kTileSizeBytes`);
`UpdateTileMappings` maps or (with an invalid heap) unmaps tiles on a queue;
`TileMappingOf` reads a mapping back; a tiling query (`TextureTilingOf`) returns the driver's
`TextureTiling` (standard mips as tile grids, small mips packed). A batch is
validated whole before any tile changes. Reads of unmapped tiles return zero
and writes are discarded. It is implemented on D3D12, the null backend and the
validating device.

**`GpuTexturePool`** (`gpu_texture_pool.h`) is a `TextureUploadSink`. It
creates one reserved texture per streamed texture and one tile heap for the
streamer's page pool, page N being heap tile N. An upload maps the level's
tiles onto its pages, repacks data at the API's 512-byte offsets and 256-byte
row pitches into a rotating per-frame upload buffer, and records the copy. An
eviction unmaps the level. Mapping updates run on the same queue as copies and
draws, which is what makes immediate page reuse safe. Per frame: `BeginFrame`,
`TextureStreamer::Update(pool)`, `EndFrame`, then draw with
`MinLOD = ResidentLevel`.

**Verified** on the HD Graphics 530 and WARP: data uploaded through mapped
tiles reads back exactly, unmapped tiles read as zero, streamed bytes read back
exactly through the pool, and evicted levels read as zero. A fix came out of
this work: command lists were destroyed before the GPU had run them.

## Virtual texturing

`VirtualTexture` (`virtual_texture.h`) presents one image far larger than video
memory as bordered pages in a physical cache. Each mip has a page table whose
entries always point at the finest resident page covering them, falling back to
an ancestor, and the top-mip page is pinned so a lookup always resolves. A
border on every page keeps filtering from reading a neighbour. Samplers report
wanted pages with `RequestPage` / `RequestSample`, safe from any thread
including pixel shaders on job threads; `Update` turns requests into loads,
coarsest first, on IO threads, copies at most `maxUploadsPerFrame` pages in, and
evicts least-recently-used pages the frame did not touch. `Sample` performs the
lookup on the CPU.

`samples/19_virtual_texture` flies over a 65,536 x 65,536 texture (21 GiB with
mips) with a 41 MiB cache: 98.7% of pixels at the exact detail requested, worst
update 0.83 ms. Seven tests, stable over 8 runs.

## GPU virtual texturing

`GpuVirtualTexture` (`gpu_virtual_texture.h`) mirrors a `VirtualTexture` into
two RHI textures: the page table as one R32_UINT texture whose mip m is virtual
mip m's table (entries in `EncodePageTableEntry` form), and the cache atlas as
RGBA8. `Sync`, called after `Update`, uploads only what changed — the dirty
page-table mips and the cache pages the update wrote — through a rotating set
of upload buffers guarded by one fence, and the first call uploads every table
mip plus the pinned pages. `ShaderSource()` returns the HLSL: `VtParams` and
`VtSample(p, uv, lod)`, which does the same page lookup, ancestor fallback and
bordered bilinear filter as `VirtualTexture::Sample`, with the same float
operations in the same order.

`Flush` runs several internal updates; its dirty lists now accumulate across
them, so one `Sync` after `Flush` mirrors the whole flush. (Before this, a
mirror that synced after `Flush` missed every page loaded by all but the last
internal update.)

**Verification** (`renderer_gpu_virtual_texture`): 2,048 random (u, v, lod)
samples run through a compute shader on every Direct3D 12 adapter and are
compared with `VirtualTexture::Sample`: while only the pinned page is resident
(every lookup an ancestor fallback), while pages stream in a few per frame under
a cache smaller than the request set, and after a flush. Result on the HD
Graphics 530 and on WARP: zero samples differ in all three rounds. A Null
backend test checks what `Sync` uploads and when.

## GPU-rendered streaming

`renderer_gpu_streaming_render` streams six 1024 x 1024 RGBA8 textures through a
tile heap a quarter the size of their full chains while a compute shader
samples every texture each frame at the level the frame wants, clamped to the
resident level, through a view starting at that level. 92,160 samples per
adapter are compared texel for texel with the pattern the sources generate: zero
differ on the HD Graphics 530 and WARP, the sharpest level reached is 0, the
heap is oversubscribed so levels are evicted and reloaded, and the streaming
work for a frame (update, tile mapping, copies, submit) peaked between about 4 and
9 ms across runs on the HD Graphics 530 and WARP, with the 99th-percentile frame asserted under 50 ms (the single worst frame
is reported, not asserted: one 182 ms scheduler hiccup was seen under parallel
tests).

## GPU virtual texture feedback

A virtual texture needs to be told which pages the view samples. On the CPU a
sampler calls `RequestSample`; a GPU renderer cannot, so the sampling shader
records the request itself. `GpuVirtualTextureFeedback`
(`gpu_virtual_texture_feedback.h`) is that mechanism, the one production virtual
texturing systems use on every GPU:

1. `Begin(list)` clears a counter buffer (one 32-bit counter per page of every
   mip, in the CPU's flat page order) by copying from a zero buffer.
2. The frame's shaders call `VtFeedback(p, uv, lod)` next to `VtSample`, which
   picks the same page `RequestSample` would and does `InterlockedAdd` on its
   counter. The counters also say how many samples each page got.
3. `End(list, fence, value)` copies the counters into the next slot of a
   readback ring; the caller submits and signals the fence.
4. `Resolve()` polls without blocking, reads back every finished frame oldest
   first and calls `RequestPage` for each non-zero counter. If every slot is in
   flight, `End` waits for the oldest. The latency is the frames in flight;
   requests of an older frame are as good as new ones, since pages stay
   requested until the streamer has them.

**Why not D3D12 sampler feedback.** Hardware sampler feedback needs a
`SamplerFeedbackTier` of 0.9 or higher (`D3D12_FEATURE_DATA_D3D12_OPTIONS7`). A
probe of both local adapters reports tier 0 on the Intel HD Graphics 530 and on
WARP (tiled resources tier 3, highest shader model 6.5 and 6.2). The
shader-written buffer works on any backend with storage buffers and atomics, so
it is the path that runs here and is verified here, and it is not lesser in
quality: it counts samples as well as pages, which the hardware feedback maps do
not. Where sampler feedback exists it would feed the same `RequestPage` calls;
nothing above this class would change.

**Verification** (`renderer_gpu_virtual_texture_feedback`, Exhaustive validation,
HD Graphics 530 and WARP):

- *Counters.* 2,048 random (u, v, lod) samples run through a compute shader; the
  counters read back match, page for page, the counts computed on the CPU with
  `RequestSample`'s arithmetic: 0 of 1,365 pages differ, 2,048 samples counted,
  601 distinct pages, and the texture's next `Update` sees exactly those 601
  requests though no CPU sampler asked for any.
- *Closed loop.* 256 samples of a small region at four levels run each frame
  with feedback on the GPU; each frame's counters are resolved, streamed in and
  mirrored by `GpuVirtualTexture::Sync`. Every sample reaches the level it asked
  for after 12 frames, starting from none (the pinned page only), and the GPU
  result then equals `VirtualTexture::Sample` (0 of 256 differ).
- The Null backend checks the ring: a frame is not read before its fence, a full
  ring waits for the oldest slot, and nothing leaks.

## BC6H

`DecodeBc6hBlock` / `EncodeBc6hBlock` (and `CompressBc6hSurface`,
`DecompressBc6hSurface`) handle BC6H as half-float RGB, 16 bytes per 4x4 block.
Texels travel as half-float bit patterns (`FloatToHalf`, `HalfToFloat`, which
round to nearest even and round-trip every non-NaN pattern, checked in a test).
The RHI gained `Format::BC6HUfloat` and `BC6HSfloat`; `TextureFormat::Bc6hRgb`
is the unsigned one.

**Decoder.** All 14 modes (ten two-subset modes with 3-bit indices and 32
partitions; four single-subset modes with 4-bit indices), the endpoint
transforms, unsigned and signed unquantization, and the 6-bit-weight
interpolation, from one table of bit layouts that the decoder and encoder share
(a `static_assert` checks that every layout accounts for all 82 or 65 header
bits). Reserved modes decode to zero.

**The decoder is checked against the hardware.** `renderer_bc6h_hardware` uploads
64 random blocks of each of the 14 modes plus 256 blocks the encoder produced,
for both formats, as BC6H textures, loads every texel in a compute shader on each
adapter, and compares with the CPU decoder bit for bit: 0 of 1,152 blocks differ
for unsigned and for signed, on the HD Graphics 530 and on WARP. It found one
wrong layout (mode 5), which was read straight out of the hardware: with only
mode 5's bits set one at a time, each bit changed one channel of one endpoint by
an amount that identifies its position in the field.

**Encoder.** Texels are searched in the integer domain the decoder interpolates
in: the half bit pattern scaled by 64/31 (unsigned) or 32/31 (signed), where
squared error is a relative, roughly logarithmic, error in the output. Per
subset the endpoints come from the principal axis refined by alternating index
assignment and a 2x2 least-squares solve; they are quantized to each mode's
width, the delta transform and its range are applied (including the wrap-around
of the sum), and indices are re-assigned against the exact palette the decoder
will build, with anchors restricted to the lower half. Fast tries modes 11 and
12. Normal adds modes 13 and 14 and the four best-ranked partitions of the ten
two-subset modes, then polishes the two best candidates by coordinate descent on
the endpoint codes (each moves one or two steps while the error falls and the
mode can still store it). High tries all 32 partitions and polishes four.

| Variant | Quality | Measured | Floor |
|---|---|---|---|
| UF16, tone-mapped | Fast | 35.10 dB | 34.5 dB |
| UF16, tone-mapped | Normal | 38.48 dB | 38.0 dB |
| UF16, tone-mapped | High | 38.49 dB | 38.0 dB |
| SF16, tone-mapped | Normal | 38.34 dB | 37.8 dB |

The test image spans 0.001 to 200 in half floats (gradients, hard stripes, a
colour ring with a wide brightness ramp, noise); PSNR is taken after
`x / (1 + |x|)` so dark detail and highlights count alike. Infinity and NaN clamp
to the largest finite half, negative values to zero when unsigned, and a solid
block stays within a quantization step.

## WARP and constant array indices

An earlier version of this work saw WARP return element 0 for every index of a
bound texture array. The isolated reproduction (`rhi_d3d12_texture_array`) binds
six distinct textures as one `Texture2D Tex[6]` and reads one texel of each, as
FXC bytecode and as DXC DXIL, with four index styles:

| Index | HD Graphics 530 | WARP |
|---|---|---|
| literal (`Tex[3]`) | correct | element 0 only (5 of 6 wrong) |
| unrolled loop | correct | element 0 only |
| run-time loop | correct | correct |
| `NonUniformResourceIndex` | correct | correct |

The FXC output is right (`ld ... T0[3]`), DXIL fails identically, and the
hardware adapter is right for all four, so the fault is in WARP's handling of a
constant index into a descriptor range: not the RHI's descriptor writes (they are
the same for the cases that work) and not a compiler. Run-time and non-uniform
indices, which is how bindless access is written, are correct on both. The test
records the two failing combinations on the software adapter (vendor 0x1414) as
a known defect and requires everything else to be exact; if a WARP update fixes
it, the exemption can go.


## RHI defects found by this work

- D3D12 `CopyBufferToTexture` took the destination texel origin as the origin of
  the *source* box inside the buffer footprint, so any copy to a non-zero x or y
  failed at command list close. The box now starts at zero and the origin only
  places the region in the texture. Tile-sized page copies need this.
- The Null backend computed a mip's extent as `size >> mip` without clamping to
  one, so every 2D copy to a mip above 0 (depth 1 shifts to 0) was rejected as
  leaving the mip. It clamps now, for buffer-texture and texture-texture copies.
- The D3D12 backend discarded the shader compiler's messages, so a failed HLSL
  compile said only that it failed. It now logs the compiler's output.
- A reserved texture released with a tile-mapping update still queued ended the
  process under the D3D12 debug layer. The update is a queue operation naming the
  texture (and its heap), so the release now waits for the queue's mapping
  updates first (a fence signalled behind every `UpdateTileMappings`), and
  `WaitIdle` drains a queue that only mapped tiles. `rhi_d3d12_reserved` has a
  regression test (it kills the process without the fix), and the pool and
  streaming tests now run with full GPU-based validation.
