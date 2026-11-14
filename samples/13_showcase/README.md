# 13_showcase — everything, in one frame

Every feature the renderer has gained so far, composed into a single picture
rather than measured one at a time. The other samples each isolate one thing;
this one is what they add up to.

What is in the frame:

| Feature | Where it shows |
|---|---|
| PBR direct lighting | a sun, a cool directional fill, and three point lights standing in for the emissive fixtures |
| PCSS soft shadows | a 2048 x 2048 reverse-Z map; the penumbra widens with the caster's distance, so contact stays sharp |
| Image-based lighting | a six-face reflection probe captured in the room itself, prefiltered to a specular chain plus an irradiance map and a BRDF table |
| Reflection probes | the mirror sphere and the metal row read the probe with a box proxy; the matte surfaces read the same capture as a global environment |
| Roughness sweep | five metal spheres from 0.04 to 0.46, the same material otherwise |
| Textures and mips | the stone floor, tiled and mipped, at a grazing angle |
| Supersampled anti-aliasing | 4x4 samples per pixel, resolved through a Mitchell filter |
| Post-processing | ambient occlusion, depth of field, bloom, auto exposure, ACES tone mapping and a baked grading LUT |

Motion blur and the temporal resolve are the two Phase 30 passes that are not
here, because a still frame has no velocity for them to follow.
[12_post_process](../12_post_process/README.md) is where those are shown.

## Build and run

From the repository root:

```powershell
cmake --preset release
cmake --build --preset release --target cyber_sample_showcase
.\build\release\samples\13_showcase\cyber_sample_showcase.exe
```

It writes two images into the current directory, or into a prefix given as the
first argument:

| File | What it shows |
|---|---|
| `showcase.ppm` | the frame with the whole chain |
| `showcase_no_post.ppm` | the same frame with tone mapping alone |

The pair is the comparison: same geometry, same lighting, same resolve, with
and without the post chain. The run also prints what each post pass cost in
CPU milliseconds, and the exposure the frame was metered to.

## Notes on the scene

The probe sits above the metal row rather than in it, and the metals are left
out of the capture. A probe cannot reflect itself, and the row stands close
enough that including it would fill the cube with its own dark backs.

Only the objects standing in the room cast shadows. The room is a closed box:
its ceiling would occlude the sun everywhere, and its flat walls and floor only
self-shadow into acne. The shadow map's extent is sized for the whole room
rather than for the casters, because a receiver that falls outside the map has
no depth to compare against and reads as fully shadowed.

The floor is a single twelve-metre quad. Two of its triangles straddle the
near plane behind the camera, and the clipped corners project far outside the
rasterizer's addressable range. The pipeline used to reject such a triangle
whole, which left a wedge of the room with no floor in it; it now clips to the
guard band in clip space instead. This sample is what found that.
