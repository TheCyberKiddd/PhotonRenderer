# Phase 33 design — Texture compression and streaming

Date: 2026-11-14
Status: Complete for the BCn codec and the CPU side of streaming. Decoders for
BC1, BC4, BC5 and BC7 define the reference output; encoders for the same four
formats are held to PSNR floors in CI. The residency manager streams mips into
a fixed page budget on background threads and is exercised by a sample with
16.7 times more texture data than its budget. An experimental CPU virtual
texture cache and sample are also implemented. Binding either page cache to
D3D12 reserved resources and GPU sampler feedback are not implemented.
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

- `cyber/assets/bc_codec.h`: block and surface decoders and encoders for BC1,
  BC4, BC5 and BC7, mip-chain generation (linear or sRGB-correct), and PSNR.
- `cyber/renderer/texture_streaming.h`: `TextureStreamer` (the residency
  manager, page pool, IO threads, eviction), `ComputeWantedMip`, and
  `RenderResidencyView`.
- `cyber/platform/thread.h`: `Thread::SetCurrentThreadPriority`, so IO threads
  run below the frame.
- `samples/18_texture_streaming`: an oversubscribed corridor flythrough with the
  hitch, budget, and lag measurements, and residency-view images.

Out of scope, and stated rather than implied:

- **D3D12 reserved resources.** The streamer allocates pages and tells a
  `TextureUploadSink` which pages each level occupies; a D3D12 sink would map
  them with `UpdateTileMappings` and copy through the upload ring. The RHI has
  no tiled-resource API yet, and the sink interface was shaped so adding one
  does not change the streamer.
- **BC2, BC3, BC6H encoders.** The containers already carry them (Phase 9); the
  roadmap names BC1, BC4, BC5 and BC7, which cover colour, single-channel,
  normal maps and high-quality colour with alpha. BC6H (HDR) is the next one
  worth writing.
- **Hardware virtual texturing.** The CPU reference cache and sample exercise
  page-table lookup, feedback, and residency, but the page cache is not bound to
  D3D12 reserved resources and does not consume GPU sampler feedback.

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
- **BC7**: opaque blocks try mode 6, then modes 1 and 3 on the partitions a
  cheap estimate ranks best, and mode 0 on three-subset partitions at High.
  Blocks with alpha try modes 6 and 5. The partition estimate is the summed
  line-fit residual of each subset — total variance minus variance along the
  principal axis — which costs a 3x3 power iteration per subset instead of a
  full encode. Solid blocks use a mode 5 single-colour table (7-bit endpoints
  at weight 21/64 reach every 8-bit value), so flat colour is exact; mode 6
  cannot promise that because its p-bit is shared by an endpoint's channels.
  P-bits are chosen per endpoint or per pair by quantization error, and
  exhaustively at High.

`BcQuality` sets the refinement passes and, for BC7, how many partitions are
tried: Fast 1, Normal 4, High 12 plus four three-subset ones. Quality never
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
| BC7 (RGBA, alpha ramp) | Normal | 40.08 dB | 39.5 dB |

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

- The page pool is not yet backed by D3D12 reserved resources; see Scope.
- BC7 does not search modes 2, 4 and 7, or mode 4/5 channel rotation. Mode 7
  would most help blocks with sharp alpha edges, which is the 40 dB case.
- Streaming moves one level per texture at a time. That keeps chains
  contiguous and the logic simple; a texture that needs several levels at once
  fills them over consecutive reads, not in parallel.
- Wanted mips come from analytic screen coverage. GPU sampler feedback
  (D3D12 sampler feedback maps) would be exact for hidden or partly covered
  surfaces; it needs a capable adapter, which the HD Graphics 530 is not.
