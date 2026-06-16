# Upstream VIP-Sim patches

Patched copies of VIP-Sim shaders/scripts kept here for traceability. Each
file in this directory is a **modified** version of the corresponding upstream
file under `Assets/VisualEffects/` in the Unity project. Patches are applied
when bugs in upstream prevent clean pixel-fidelity validation; the goal is to
keep the modification visible, documented, and replayable on another machine.

VIP-Sim is licensed CC BY 4.0; modifications are permitted with attribution.

## How to apply (on a fresh Windows Unity checkout)

Copy each file in this directory over the corresponding upstream path:

```bash
cp unity_helpers/upstream_patches/FovealDarkness.shader \
   "/mnt/c/Users/MI-Pool 8/Documents/ws25-26/max-projekt/VIP-Sim/windows/Assets/VisualEffects/Shaders/FovealDarkness.shader"
```

## Patches

### `FovealDarkness.shader`

**Bug**: in the original upstream shader, line 78 reads
`fixed4 color = tex2D(_MainTex, adjustedUV);`. `adjustedUV` is `uv` with
`uv.x` scaled by the aspect ratio (`_ScreenWidth / _ScreenHeight`). It is only
meant for the distance/centre computation a few lines earlier
(`delta = adjustedUV - adjustedMousePos`). Using it for the texture sample
means any `uv.x > 1/aspectRatio` samples outside `[0, 1]` and produces
clamp/wrap artefacts. At our 2:1 capture aspect, the right half of every
foveal-darkness render is corrupted stripes; at 16:9 (the typical game
aspect) any `uv.x > 9/16 ≈ 0.56` is corrupted too.

**Fix**: one-line change — `tex2D(_MainTex, adjustedUV)` → `tex2D(_MainTex, uv)`.
`adjustedUV` is still used correctly for the distance math above.

**Discovered / applied**: 2026-05-13 during AVP pixel-fidelity validation
(see `docs/tier2_unity_capture_pipeline.md`).
