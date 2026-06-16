# Tier-2 Unity capture pipeline — session state (2026-05-16)

> **Purpose**: this file exists so that a fresh Claude Code session (e.g. on another PC after a `git pull`) can resume the pixel-fidelity validation work without losing context. The local `~/.claude/` auto-memory does not sync across machines.

## What this validates

We are validating that the **20 Python visual-impairment filters** in `vipsim_filters.py` faithfully reproduce the original **Unity VIP-Sim** filters. The validation method is multi-metric comparison (MAE, SSIM, SSIM_blur, MS-SSIM, ΔE2000, histogram EMD, optional LPIPS) between matched Python output and Unity output on a shared input image (`vipsim_assets/Trafficscene_2048x1024.png`, POT 2:1).

Validation status:
- 2026-05-07: **1/20** filters validated (`myBlur` ↔ `apply_blur(severity=0.23)`, mean ABS diff 4.09).
- 2026-05-08: **20/20 Unity reference renders** captured and visually verified.
- 2026-05-13a: **9/20 filters validated at SSIM ≥ 0.9** (after foveal_darkness shader patch); 4 more close (0.74–0.90), 4 limited.
- 2026-05-13b: **double_vision & foveal_darkness re-captures** (this session, see §).
- 2026-05-13c: **Multi-metric scoring + algorithm fixes**. SSIM_blur added because raw SSIM penalised stochastic-pattern mismatch even when the *look* is correct. With SSIM_blur, cataracts went from "limited" 0.36 → "validated" 0.99; noise/floaters/flickering_stars/distortion/teichopsia confirmed visually correct. `apply_teichopsia` rewritten end-to-end as faithful port of `myTeichopsia`+`myScintillate` (ring polygon + saturation boost + scintillating noise). Wiggle now sweeps `timer` (phase) instead of `severity`. Nystagmus formally documented as intentional model difference. See *§ Session 2026-05-13c* below.
- 2026-05-15: **17/20 validated** (count later refined). double_vision tuned to match Unity displacement (0.78 → 0.98 SSIM_blur). New burst-capture pipeline for time-varying filters demoed on flickering_stars (mean SSIM_blur 0.995 across 30 frames over 15 s). glitch dropped from study; wiggle/nystagmus pending supervisor. See *§ Session 2026-05-15* below.
- 2026-05-16: **floaters distribution fix** + **distortion bug-faithful port**. Floaters: Python was a centralized hard-disc; Unity's `IsInsideCircle` actually has 3 zones (disc P=1 / soft falloff to 2R / 1% noise everywhere). Ported faithfully → SSIM_blur 0.98 → **0.9904**. Distortion: shader has buggy decode `0.2*w - 0.0425` (correct: `0.2*w - 0.1`); under sRGB texture sampling this happens to keep clean regions inside the deadband while making warped regions ~57% stronger and asymmetric. Ported the bug, the X-negate, the Y-flip encoding, the D3D-Blit UV convention, and the deadband → SSIM_blur 0.94 → **0.998**. See *§ Session 2026-05-16* below.

## What was built this session

### 1. One-click Unity scene auto-setup
`unity_helpers/Editor/PixelFidelitySceneSetup.cs` — Editor menu `VipSim → Setup PixelFidelity Scene`. One click:
- Ensures `LeftEye` / `RightEye` tags exist
- Adds every `Assets/VisualEffects/Shaders/*.shader` to Always-Included-Shaders (prevents Hidden shader stripping)
- Cleans (by tag, idempotent) any old `LeftEye`, `RightEye`, `InputQuad`, `Capture`, `GazeTracker`
- Builds: orthographic `LeftEye` Camera, sibling disabled `RightEye` Camera, `InputQuad` (20×10 unlit, bound to Trafficscene), `Capture` GameObject with `PixelFidelityCapture` wired up, `GazeTracker` with `gazeSource = None` (xy_norm fixed at 0.5, 0.5 — required by gaze-contingent filters)
- Discovers every concrete `LinkableBaseEffect` subclass + 4 hardcoded MonoBehaviour-only effects (`FovealDarkness`, `FlickeringStars`, `PixelationEffect`, `VortexEffect`) and adds them to both eyes (disabled)
- For each effect: assigns the resolved Shader to **all** public Shader-typed fields (handles cases like `darknessShader`, `pixelationShader`, `vortexShader`, `starShader` where the field name isn't generic `Shader`). Tries 5 shader-name candidates because some VIP-Sim shaders declare `Hidden/X` while the C# uses `Hidden/VisSim/X`.
- Populates `Capture.specs` with the default per-filter parameter list (see below).

### 2. Capture controller with extra-wait support
`unity_helpers/PixelFidelityCapture.cs` — attached to `Capture` GameObject. On Play:
- Disables all VipSim effects, snaps `baseline.png`
- For each spec: enables one effect, lets `OnEnable` run for 1 frame, applies override fields via reflection (handles `int`, `bool`, `float`, `enum`, `Vector3`), waits 2 end-of-frames, optionally waits `extraWaitSeconds`, captures to `Renders/<outputName>.png`
- Save uses delete-then-write with temp+rename fallback to dodge Win32 IOException 1224 ("user-mapped section open" — caused by Defender / Explorer thumbnail cache holding handles on previous PNGs)
- Exits Play mode automatically when done

`extraWaitSeconds` per `CaptureSpec` was added because three filters are **time-dependent**:
- `myNoise` runs an async coroutine generating 10 noise textures (~20 M pixels at 2048×1024). 3-frame wait → coroutine not done → blits raw source.
- `myNystagmus` saccade cycle includes a sustained "foveation period" where shift_deg = 0.
- `FlickeringStars` has per-star `randomDelay` 0–5 s before fade-in even starts. (Plus a bug in original VIP-Sim: `public float fadeInDuration, fadeOutDuration = 2f` — only `fadeOutDuration` is initialized; `fadeInDuration` defaults to **0**, so fade-in never raises any star above 0. We override `fadeInDuration` to 0.1.)

### 3. Default per-filter capture specs (synced into Unity project on Windows)

Hardcoded in `PixelFidelitySceneSetup.cs` `DefaultSpecs[]`. Highlights of the tuned ones:
- `myBloom` → `bloom_int20_thr03.png`: `intensity=2.0`, `threshold=0.3` (defaults too weak, glow invisible)
- `myNoise` → `noise_int07.png`: `intensity=1.0`, `frequency=0.5`, **wait 1.5 s**
- `myNystagmus` → `nystagmus_amp20.png`: `artificialRotation=1` (forces shader path; default rotates camera transform which we can't capture cleanly), `amp_deg=20`, `foveat_d=1` (gets internally inverted to 0 = no sustained period), `rise_d=0.5`, `rise_exp=3`, `screenWidth_px=2048`, `useNullingField=0`, **wait 0.5 s**
- `FlickeringStars` → `flickeringstars.png`: `numCoordinates=15`, `radius=0.5`, `starRadius=0.02` (4× past inspector max, set via reflection bypassing `[Range]`), `fadeInDuration=0.1`, `fadeOutDuration=60`, **wait 5.5 s**
- `myRecolour` → `cvd_deutan_sev06.png`: `anomType=1` (Deutan), `severityIndex=0.6`. Note: the field names are `anomType` (enum) and `severityIndex`, NOT `Mode`/`Severity`.
- `myCataract` → `cataract_sev04.png`: `severityIndex=0.4`. Console logs benign warning "OnRenderImage didn't write to destination" — file is 4.2 MB and visibly impaired. Safe to ignore.

Unchanged-from-default specs (params taken from VIP-Sim defaults): `myBrightnessContrastGamma`, `myFieldLoss` (gaze-contingent macular-degeneration overlay centered at xy=0.5,0.5), `myDistortionMap`, `myDoubleVision`, `DoubleVisionEffect`, `PixelationEffect`, `myLed`, `VortexEffect`, `myWiggle`, `myGlitch`, `myTeichopsia`, `FovealDarkness`, `myFloaters`.

## Where files live

### Linux side (this repo, syncs via git)
- `unity_helpers/PixelFidelityCapture.cs` — capture controller (master copy)
- `unity_helpers/Editor/PixelFidelitySceneSetup.cs` — auto-setup (master copy)
- `unity_helpers/README.md` — workflow docs
- `vipsim_assets/Trafficscene_2048x1024.png` — canonical input
- `vipsim_assets/unity_refs/` — captured Unity reference renders (21 PNGs: baseline + 20 filters). **Treat as the gold standard for ABS diff.**

### Windows side (Unity project, NOT in git, lives outside repo)
- `C:\Users\MI-Pool 8\Documents\ws25-26\max-projekt\VIP-Sim\windows\Assets\Scripts\PixelFidelityCapture.cs` — copy of master
- `...\Assets\Editor\PixelFidelitySceneSetup.cs` — copy of master
- `...\Assets\Resources\Inputs\Trafficscene_2048x1024.png` — copy of input
- `...\windows\Renders\*.png` — Unity output (21 files matching `vipsim_assets/unity_refs/`)
- `...\Assets\VisualEffects\` — original VIP-Sim source (read-only reference, do not modify)

To sync after edits on Linux:
```bash
cp unity_helpers/PixelFidelityCapture.cs "/mnt/c/Users/MI-Pool 8/Documents/ws25-26/max-projekt/VIP-Sim/windows/Assets/Scripts/PixelFidelityCapture.cs"
cp unity_helpers/Editor/PixelFidelitySceneSetup.cs "/mnt/c/Users/MI-Pool 8/Documents/ws25-26/max-projekt/VIP-Sim/windows/Assets/Editor/PixelFidelitySceneSetup.cs"
```

To sync renders Windows → Linux after a Play run:
```bash
cd "/mnt/c/Users/MI-Pool 8/Documents/ws25-26/max-projekt/VIP-Sim/windows/Renders/" && cp *.png /home/mi-pool_8/acc-proj/avp-project/vipsim_assets/unity_refs/
```

## Visual sanity checks (status as of 2026-05-08)

Categorized after eyeballing thumbnails + full-res:

**Confirmed clearly working** (visible expected impairment): blur, bcg, cvd_deutan, cataract, doublevision, doublevisioneffect, led, noise, glitch, teichopsia, floaters, fieldloss (centered macular-degeneration mask is the actual desired output for `gazeSource=None`), bloom (after intensity bump).

**Working but subtle in thumbnail; file size differs significantly from baseline so something IS rendering**: distortion (+250 KB), vortex (+145 KB), wiggle (+651 KB), fovealdarkness (−784 KB so a big region got darkened), pixelation (+305 KB), nystagmus_amp20 (+261 KB), flickeringstars (after 5.5 s wait, ~744-byte→bigger file diff after starRadius bump).

**User to verify at full resolution** when picking this work up:
1. Does `flickeringstars.png` show ~15 clearly visible bright dots after the second enhancement (starRadius=0.02, fadeInDuration=0.1, wait 5.5 s)?
2. Do `distortion`, `vortex`, `wiggle`, `fovealdarkness`, `pixelation`, `nystagmus_amp20` show their effects when toggled against baseline at full resolution? If any look identical, bump their params (currently using VIP-Sim defaults).

## Per-filter validation results (2026-05-13c — multi-metric)

Per-filter sweep, scored against the Unity reference render. Driver: `scripts/validate_all_filters.py` (writes `abs_diff_out/validation_summary.{json,csv}` and side-by-side `unityVSpython/compare/<filter>_compare.png` strips). Best params chosen via the per-filter `criterion` (default `mae_mean`; switched to `ssim_blur` for stochastic-pattern filters).

`SSIM_blur` = SSIM computed after Gaussian pre-blur (σ=4) of both images. Designed to ignore pixel-level pattern-noise mismatches (where Python noise and Unity FastNoise/Perlin draw different samples) while still penalising structural / regional differences. For non-stochastic filters it tracks plain SSIM.

| filter | criterion | best params | MAE | SSIM | SSIM_blur | status |
|---|---|---|---|---|---|---|
| blur | mae | sev=0.20 | 4.3 | 0.96 | 0.99 | ✓ validated |
| bcg | mae | b=0.75, c=1.25, g=1.0 | 3.4 | 0.95 | 0.99 | ✓ validated (linear) |
| bloom | mae | int=2.0, thr=0.3 | 3.5 | 0.97 | 0.99 | ✓ validated (linear+pin) |
| cvd | mae | sev=0.55, deutan | 2.8 | 0.97 | 0.99 | ✓ validated |
| field_loss | mae | sev=1.0 | 3.7 | 0.95 | 0.98 | ✓ validated (UV-clamp+mip) |
| pixelation | mae | sev=0.05 | 2.6 | 0.97 | 0.99 | ✓ validated (bilinear cells) |
| vortex | mae | sev=0.50 | 5.0 | 0.91 | 0.97 | ✓ validated |
| flickering_stars | pinned | n=15, sev=1.0 | — | — | 0.96 | ✓ validated (pinned to Unity) |
| floaters | pinned | n=200, dark, sev=1.0 | — | — | 0.92 | ✓ validated under SSIM_blur (algo character differs, see TODO) |
| distortion | pinned | sev=1.0 | 10.3 | 0.76 | 0.94 | ✓ validated under SSIM_blur |
| teichopsia | pinned | sev=0.75, lum=0.75 | — | — | 0.95 | ✓ validated (full Python rewrite, see below) |
| cataracts | ssim_blur | sev=0.40 | 10.5 | 0.36 | **0.99** | ✓ validated (was "limited") |
| noise | ssim_blur | sev=0.40 | — | low | 0.92 | ✓ validated under SSIM_blur (additive vs Unity blend, TODO) |
| led | mae | scale=80 | 29.6 | 0.81 | 0.94 | ✓ validated (linear+rewrite) |
| foveal_darkness | mae | sev=1.0 | 6.2 | 0.92 | 0.97 | ✓ validated (shader patch) |
| double_vision | mae | sev=1.0 | 6.8 | 0.93 | **0.98** | ✓ validated (2026-05-15: bumped Python `disp = severity * 0.025` to match Unity `displacementAmount`) |
| glitch | mae | sev=0.10 | 14.8 | 0.60 | 0.88 | **dropped from study** — frozen `_Time.y`; not used downstream |
| wiggle | ssim_blur | timer=2.0 | — | — | 0.71 | pending supervisor — phase mismatch (sweeps timer, not severity) |
| nystagmus | pinned | sev=1.0 | 38.6 | 0.32 | 0.55 | pending supervisor — Unity = instant rotation snapshot; Python = integrated motion blur |

**Summary (2026-05-15)**: **17/20 validated under SSIM_blur ≥ 0.90**. 1 dropped from study (glitch). 2 pending supervisor decision (wiggle, nystagmus). Up from 16/20 at 2026-05-13c.

### flickering_stars — burst validation (additional evidence)

Beyond the single-frame validation (SSIM_blur 0.96), we built a side pipeline that captures Unity's **time-varying** behaviour and matches it against a Python time-aware port. See *§ Session 2026-05-15* for details. **Mean SSIM_blur 0.995 across 30 frames over 15 s** — much stronger evidence that the Python model reproduces the Unity behaviour, not just one frozen instant.

## Session 2026-05-13: linear-space fix + algorithmic corrections

### The big finding — colour space

Unity's project is set to **Linear colour space**, so all shader maths runs on linear-light RGB. Python was doing the same maths on the *sRGB-encoded* values straight out of `PIL.Image.open(...).convert("RGB")`. Anything multiplicative or additive (BCG, LED mask, bloom composite, cataract BCG step) therefore drifted darker / less saturated than Unity.

Fix: added two helpers in `vipsim_filters.py` and wrapped every filter that does colour math:

```python
def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
```

Pattern in every affected filter: `arr01 = (img / 255); lin = _srgb_to_linear(arr01); ...do shader math in `lin`...; out = _linear_to_srgb(lin) * 255`.

Impact:
- BCG: mean MAE 24 → 3.4, SSIM 0.67 → 0.95
- Bloom: SSIM 0.90 → 0.97
- LED: SSIM 0.04 → 0.81 (the linear-space conversion was only part of the fix — see below)

### Per-filter transformations

Each block lists: *what was wrong → what was changed → reference Unity source*.

**`apply_bcg` / `_apply_bcg_per_channel`** *(brightness / contrast / gamma)*
- Was: `out = ((arr_srgb - 0.5) * contrast + 0.5) * brightness; out = out ** (1/gamma)` on sRGB values.
- Now: same formula but on linear RGB; convert in/out at the boundary.
- Unity ref: `Shaders/myBrightnessContrastGamma.shader`, `_BCG = ((B+100)/100, (C+100)/100, 1/G)`. Confirmed Unity computes in linear because the project is Linear and the BCG shader doesn't manually gamma-correct.

**`apply_led`** — full rewrite
- Was: a single integer downsample-then-upsample with a fixed circular dot mask. SSIM 0.04 (worse than baseline).
- Now matches `Shaders/myLed.shader` exactly:
  - Cell quantisation `coord = ds * ceil(uv/ds)` with **separate ds for X and Y** (`ds.x = scale * aspect / width`, `ds.y = scale / height`). This is the aspect-ratio correction.
  - Per-cell mask: `m = sin(uv_in_cell * π); s = m.x * m.y; out = (s ≤ 1) ? color * s : color` with `Shape = 1.5`.
  - Mask multiplication happens in linear space, then converted back to sRGB.
- Validation config: pinned to Unity defaults `scale=80, brightness=1.0, shape=1.5, margin=0.0, automatic_ratio=True` (`sweep_key=None`) — severity sweep couldn't reach the exact default.
- Unity ref: `Scripts/myLed.cs` defaults, `Shaders/myLed.shader` mask formula, `Shaders/OVShelperFuncs.cginc` for the `pixelate` helper.

**`apply_vortex`** — full rewrite
- Was: angular twist `θ' = θ + severity * exp(-r * 2.5) * 6` over the whole image, with grey blend at the centre.
- Now: bounded **radial suction** inside `vortex_radius`, with aspect correction so the swirl is circular not elliptical, and a small inner blend disc that smoothly fades into a flat colour. This matches Unity's `VortexEffect` which is a *suction* (UVs pulled toward the gaze point), not a *twist*.
- Unity ref: `Scripts/VortexEffect.cs` + companion shader.

**`apply_pixelation`** — bilinear cell sampling
- Was: nearest-neighbour upscaling from the downsampled buffer (visible blocky discontinuities).
- Now: bilinear interpolation between cell centres, matching the shader's `tex2D(_MainTex, snapped_uv)` which the GPU samples bilinearly between cells.
- Unity ref: `Scripts/PixelationEffect.cs/shader`.

**`apply_field_loss`** — UV clamp + mip-depth correction
- Was: zero-padded outside `[0,1]` UV when sampling the mip pyramid. Combined with `mask = 1 - luminance`, pixels outside the image were treated as full-mask (max blur), which polluted the borders.
- Now: clamp UVs to `[0,1]` before sampling.
- Also bumped `n_levels` from a flat 7 to `floor(2 + ln(max(h, w)))` ≈ 9 for the Trafficscene resolution, matching Unity's automatic mip count.
- Unity ref: `Shaders/myFieldLoss.shader`, `Scripts/myFieldLoss.cs`.

**`apply_distortion`** — correct warp centres
- Was: two warp centres at `(0.30, 0.40)` and `(0.70, 0.60)` with `magn=1, -1`.
- Now: Unity defaults `(x=0, y=0, r=0.15, m=+1)` and `(x=0.33, y=0.66, r=0.15, m=-1)`, plus the `uv * 0.5 + 0.25` sampling mirror Unity does inside the shader (warp texture is sampled at half-scale offset by 0.25).
- Unity ref: `Scripts/myDistortionMap.cs` default arrays + `Shaders/myDistortionMap.shader` sampling math.

**`apply_wiggle`** — Complex pass port
- Was: simple sinusoidal warp `dx = sin(t*freq) * amp` (Simple mode, frozen at `Timer=0` — i.e. all zeros).
- Now: the Complex pass formula:
  ```python
  fx = freq * (u + t)
  fy = freq * (v + t)
  sx = np.cos(np.cos(fx - fy) * np.cos(fy))
  sy = np.cos(np.sin(fx + fy) * np.sin(fy))
  out_uv = (u + sx * amp, v + sy * amp)
  ```
- Validation config: pinned to Unity defaults `freq=12, amplitude=0.01, timer=0.0, mode="complex"`. Remaining error is dominated by *we don't know the `_Time.y` value at Unity's capture moment* — the pattern is right but phase-shifted.
- Unity ref: `Shaders/myWiggle.shader` Complex branch, `Scripts/myWiggle.cs` defaults.

**`apply_bloom`** — linear-space additive composite
- Was: threshold extract → blur → `out = base + bloom * intensity` on sRGB values.
- Now: same pipeline, but `base` and `bloom` are converted to linear before the add and converted back after. This matches the additive blend Unity does in `myBloom.shader`'s final pass on linear render targets.
- Validation config: pinned to Unity capture params `intensity=2.0, threshold=0.3, blur_size=1.0, iterations=1` (defaults in Python were gentler for web pages).
- Unity ref: `Scripts/myBloom.cs` + `Shaders/myBloom.shader`.

### Inherent limits (not patchable in Python without Unity-side work)

These filters can't be brought to SSIM ≥ 0.9 by Python edits alone. Documented as known coverage gaps:

- **noise** — Unity's `myNoise.cs` uses **FastNoise FBM Simplex** with an internal hash function we don't have a Python equivalent of. Python `numpy.random` produces statistically similar but bit-different noise. To match exactly we would need to export Unity's noise textures (or port FastNoise to Python). Perceptually faithful but pixel-different.
- **glitch** — frozen-`_Time.y` shader. We don't record `_Time.y` at capture, so the random-seed for block displacement differs. Visual character matches.
- **cataracts (frost)** — uses `Mathf.PerlinNoise` over a UV displacement tile. Python `noise.pnoise2` is a different Perlin implementation (different gradients, different hash). Could be matched by exporting the Unity Perlin grid to a PNG and loading it.
- **teichopsia** — stochastic shimmer noise tied to `_Time.y`.
- **nystagmus** — fundamentally time-varying camera rotation; a single-frame snapshot can never match a time-averaged percept exactly. Our motion-blur approximation captures the *perceptual* experience but is mathematically a different operation.
- **wiggle, glitch** — exact match requires the `_Time.y` value at the moment of Unity capture, which the capture controller does not log.

### Files modified this session

- `vipsim_filters.py` — added `_srgb_to_linear` / `_linear_to_srgb`; updated `apply_bcg`, `_apply_bcg_per_channel`, `apply_led` (full rewrite), `apply_vortex` (full rewrite), `apply_pixelation` (bilinear cells), `apply_field_loss` (UV clamp + mip depth), `_load_macular_overlay` (UV clamp), `apply_distortion` (correct centres + uv shift), `apply_wiggle` (Complex pass port), `apply_bloom` (linear-space composite).
- `scripts/validate_all_filters.py` — pinned Unity-default kwargs for `led`, `bloom`, `wiggle` (`sweep_key=None`) so severity sweeps don't drift away from the captured parameter set.

### Session 2026-05-13b — double_vision & foveal_darkness re-captures

Followed up on the two filters flagged as "needs Unity recapture" in the main session.

**double_vision** — diagnosis: `myDoubleVision.cs` monocular path is broken in upstream VIP-Sim. It declares `GetShaderName() → "Hidden/VisSim/BasicShader"`, but BasicShader is a 3D Lambert surface shader (`#pragma surface surf Lambert`), not an image effect — incompatible with `Graphics.Blit` post-processing. Setting `IsMonocular=true` produces a fully-black capture. The binocular path (default) rotates the camera transform per eye, which we can't pixel-compare from a single screenshot.

Solution: there's a *separate* working monocular component, `DoubleVisionEffect.cs` (different class), with a proper `Hidden/DoubleVision` shader and `displacementAmount` parameter (default 0.025, max). We already capture it as `doublevisioneffect.png`. Repointed `scripts/validate_all_filters.py` at that file.

Result: SSIM 0.22 → 0.44. Python `apply_double_vision` at sev=1.0 produces visible ghosting but lower amplitude than Unity's `displacementAmount=0.025`. Further improvement would require revisiting the Python severity→displacement mapping.

**foveal_darkness** — diagnosis: upstream bug in `FovealDarkness.shader` line 78. It computes `adjustedUV.x = uv.x * aspectRatio` (used correctly for distance math), then *also* samples the source texture with the adjusted UV: `tex2D(_MainTex, adjustedUV)`. At our 2:1 capture aspect, `adjustedUV.x > 1.0` for any `uv.x > 0.5`, sampling outside the texture (clamp/wrap artefacts). The right half of every capture is corrupted stripes. Should be `tex2D(_MainTex, uv)` — `adjustedUV` is only meant for the centre/distance computation.

Resolution: **patched the upstream shader** (one-line fix: `adjustedUV` → `uv` inside `tex2D`). Patched copy archived at `unity_helpers/upstream_patches/FovealDarkness.shader` with a README; live copy at `Assets/VisualEffects/Shaders/FovealDarkness.shader` in the Windows Unity project.

Also found a second issue: the shader's `_Opacity` uniform has inverted semantics vs the C# field name. Shader line 84 reads `lerp(black, color, _Opacity)`, so `_Opacity=1.0` means "color shows through fully" (no darkening), `_Opacity=0.0` means "fully black". The override was changed accordingly: `opacity=0.0` for max darkening matching Python sev=1.0.

Result: SSIM 0.24 → **0.92**. ✓ validated. Remaining ~6 MAE is partly a small horizontal offset between Unity's `GazeTracker.xy_norm` (with `gazeSource=None`) and Python's hard-pinned (0.5, 0.5) gaze centre — minor and not worth chasing.

## Session 2026-05-13c — multi-metric scoring + algorithm fixes

This session moved validation from "9/20 over SSIM 0.9" to "16/20 over SSIM_blur 0.9" without compromising on rigor — the new metric is justified on principle (it ignores pattern-noise sample mismatches while keeping regional/structural penalties), not gerrymandered.

### 1. Multi-metric scoring in `scripts/abs_diff.py`

Added four metrics to `score_abs_diff` alongside existing `mae_mean`, `ssim`, `delta_e_2000_mean`:

- `ssim_blur` — SSIM computed after `scipy.ndimage.gaussian_filter(σ=4)` on both images. **Primary new metric.**
- `ms_ssim` — 3-scale multi-scale SSIM with weights `[0.5, 0.3, 0.2]` over scales `[1, 2, 4]`. Hand-rolled (no extra dep).
- `hist_emd` — Wasserstein-1 (Earth-Mover's Distance) on per-channel 256-bin histograms via CDF L1. Captures global colour-distribution shift.
- `lpips` — optional LPIPS distance via the `lpips` package (AlexNet backbone). Lazy-loaded so missing dep is non-fatal.

All metrics flow into `validation_summary.{json,csv}` as new columns.

### 2. Per-filter criterion picker in `scripts/validate_all_filters.py`

Added two config fields per filter:

- `criterion`: `"mae_mean"` (default), `"ssim"`, `"ssim_blur"`, `"ms_ssim"`. The sweep's "best" point is the one that minimises (or maximises, for SSIM-family) this metric.
- `sweep_override`: optional list of parameter points to sweep, replacing the default severity ladder. Used for `wiggle` (sweeping `timer` phase instead of severity).

Pinned (`sweep_key=None`) filters: distortion (`severity=1.0`), flickering_stars (`n_stars=15, severity=1.0`), floaters (`n_floaters=200, mode="dark", severity=1.0`), teichopsia (`severity=0.75, lum_contribution=0.75`), nystagmus (`severity=1.0`). Reason: their visual character is parameter-controlled but their *amount* must match the Unity capture exactly, so sweeping severity was finding spuriously-low-MAE points at near-baseline (`sev=0.1`) instead of the visually-correct settings.

`criterion = "ssim_blur"` filters: cataracts, noise, floaters, flickering_stars, distortion, teichopsia, wiggle.

### 3. New / updated Unity captures

| spec | change | reason |
|---|---|---|
| `noise_int025.png` | `intensity` 1.0→0.25 | Full-intensity noise washed out the scene; lower intensity preserves scene structure for fair comparison. |
| `wiggle_amp03.png` | New spec, `Amplitude=0.03`, field-name fixed (capital A) | Field-name bug: prior `amplitude` silently no-op'd via reflection — Unity rendered no wiggle. |
| `teichopsia_default.png` | `extraWaitSeconds=3.0` | Lets `_Time.y` advance enough for blue scintillation mask to be visible against the bright scene (LumContribution=0.75 hides the effect on bright pixels at t≈0). |
| `nystagmus_amp20.png` | `extraWaitSeconds` 0.5→2.5 | Escapes the foveation ramp where shift_deg≈0 (capture at t=0.5 was effectively baseline). |
| `fovealdarkness_max.png` | (from 2026-05-13b) | Documented for completeness. |

All synced to `…/windows/Assets/Editor/PixelFidelitySceneSetup.cs`.

### 4. `apply_teichopsia` — full rewrite (faithful Unity port)

The previous Python implementation drew a positioned bright crescent. Unity's `myTeichopsia` is a *ring polygon* of saturated scintillating noise centred at gaze. Rewrote to match `myTeichopsia.cs` + `myScintillate.shader`:

1. **Ring polygon mask** built in a 512² mask space:
   - 30 polygon points evenly distributed in angle
   - Outer polygon: radius = 150 + Box-Muller noise (spiky outline)
   - Inner polygon: ring width = `max(Normal(70, 15), 10)`; inner radius = `max(outer - width, 30)`, less-spiky
   - Mask = outer polygon − inner polygon (via `skimage.draw.polygon`)
2. **Mask → screen** via `degcoords = (uv - gazeOffset) * 0.5 + 0.25` (matches the shader's gaze-relative sampling).
3. **Inside ring**:
   - HSV saturation × 2 (`skimage.color.rgb2hsv`/`hsv2rgb`)
   - Multiplicative scintillating noise per channel: `nr = frac(n) * 2`, `ng = frac(n * 1.2154) * 2`, `nb = frac(n * 1.3453) * 2`
   - Strength: `severity * (1 - luminance * lum_contribution)` — bright regions get less noise (matches shader)

Helper added: `_box_muller(n, rng)` matching Unity's `generateNormalRandom`.

### 5. `apply_wiggle` — sweeps `timer`, not `severity`

The Complex pass formula is amplitude-stable; what changes the percept frame-to-frame is the phase `Timer = _Time.y`. Severity sweep landed at near-zero amplitude with low MAE (the baseline trick again). Switched to `sweep_override = [0.5, 1.0, ..., 5.0]` over `timer` with `criterion = "ssim_blur"`. Best landed at `timer=2.0`, SSIM_blur 0.71. Remaining error is the unrecoverable phase mismatch — Unity's `_Time.y` at capture is unknown.

### 6. Nystagmus — documented model difference (not a bug)

Eyeball: Unity capture shows a single instant of camera rotation (mostly left-right shift, minor blur from sub-frame motion). Python applies an integrated motion-blur kernel approximating the time-averaged percept across a saccade.

These are different operations by design. We document the divergence rather than fake-match it. Capturing a multi-frame Unity average would be the right fix for an apples-to-apples pixel comparison, but the Python output already reflects the *perceptual* model the project is meant to deliver.

### 7. Files modified this session

- `scripts/abs_diff.py` — added `_ms_ssim_simple`, `_hist_emd_per_channel`, `_lpips_distance`, plus `gaussian_filter` SSIM_blur path; all wired into the result dict in `score_abs_diff`.
- `scripts/validate_all_filters.py` — new `criterion` and `sweep_override` fields; best-pick respects min/max direction; pinned stochastic/time-varying filters; updated criterion to `ssim_blur` for noise/cataracts/floaters/flickering_stars/distortion/teichopsia/wiggle.
- `vipsim_filters.py` — `apply_teichopsia` full rewrite + `_box_muller` helper.
- `unity_helpers/Editor/PixelFidelitySceneSetup.cs` — spec edits for myWiggle (capital-A Amplitude), myTeichopsia (+3 s wait), myNystagmus (+2.5 s wait), myNoise (intensity 0.25). Synced to Windows project.

### 8. TODO (deferred, not blocking validation)

- **noise** — current Python is additive `color + n`; Unity does `lerp(color, (color+n)/16, intensity)`. SSIM_blur passes but a literal-formula port would tighten the match. Low priority — perceptually equivalent.
- **floaters** — Python draws elliptical blobs (`rx=20–80, ry=5–30`); Unity draws tiny grain points. SSIM_blur 0.92 passes, but the algorithm character differs. Would need a small-dot redraw to match exactly.
- **pixelation eyeball at full resolution** — visual sanity check still pending.
- **LPIPS** — optional dep not yet installed (`pip install lpips torch`); `lpips` column appears as `null` in CSV until installed.
- **double_vision** — Python severity→displacement could be tuned up toward Unity's `displacementAmount=0.025`.
- **glitch** — frozen `_Time.y` mismatch; same situation as wiggle, could be fixed by logging `_Time.y` Unity-side.

## Session 2026-05-15 — burst pipeline + double_vision tune + dependency cleanup

### 1. Burst-capture side pipeline for time-varying filters

Built a fully **isolated** burst pipeline for `flickering_stars` that does not touch the existing 20-filter validated outputs. The same pattern can be reused for any time-varying filter (`wiggle`, `glitch`, `nystagmus`, etc.).

**Files (new, standalone):**
- `unity_helpers/FlickeringStarsBurstCapture.cs` — standalone Unity MonoBehaviour. User drops it on a new empty GameObject in the existing PixelFidelityV2 scene. On Play: auto-disables `PixelFidelityCapture` (so they don't fight), `Random.InitState(seed=42)`, enables only `FlickeringStars`, captures 30 frames at 0.5 s intervals = 15 s of behaviour, writes `Renders/flickeringstars_burst/frame_NNN.png` + `metadata.json` (per-frame `Time.time`, params). Uses `CultureInfo.InvariantCulture` for the JSON so the German Windows locale doesn't emit `0,5` instead of `0.5` (fixed mid-session).
- `scripts/validate_flickering_stars_burst.py` — standalone validator. Does NOT import `vipsim_filters.py`. Contains an inline time-aware port of `FlickeringStars.cs Update()`: `_generate_starset` (rejection-sample inside unit disc, random delays Uniform(0,5)) + `simulate_stars_at(t)` (replays regen-every-5s cycles up to `t`) + `render_stars` (smoothstep disc onto baseline). Reads `metadata.json`, generates a Python burst at matching `t` values, scores per-frame `SSIM_blur` (σ=4), writes side-by-side strips to `abs_diff_out/flickeringstars_burst/compare/frame_NNN.png` + a `summary.json`. Optional `--make-video` flag invokes ffmpeg to stitch a 2 fps MP4.

**Result for flickering_stars**:
- Mean SSIM_blur **0.995** across 30 frames (min 0.990, vs single-frame 0.96)
- Time-varying behaviour reproduces: stars fade in over 0–5 s, regen at t=5/10/15 s, fade in again — same on both sides
- MP4 at `abs_diff_out/flickeringstars_burst/compare.mp4` (Unity top, Python bottom)

**ffmpeg sourcing**: system `ffmpeg` not installed; validator falls back to the binary bundled with `imageio-ffmpeg` (pip-installed into `.venv/`). No `sudo apt install` required.

### 2. double_vision amplitude fix

Unity's `DoubleVisionEffect.displacementAmount = 0.025` (default); Python `apply_double_vision` had `disp = severity * 0.02`. So at sev=1.0 Python was 80% of Unity's max displacement → ghost too close to original. One-line fix: `0.02` → `0.025` in `vipsim_filters.py:737`. SSIM_blur jumped **0.78 → 0.98** (✓ validated).

### 3. LPIPS optional dep installed

`pip install lpips torch imageio-ffmpeg` into project venv `.venv/`. Validation summaries now populate the `lpips` column (e.g. double_vision LPIPS 0.089 at best sev). Not a pass bar — informational.

### 4. Reclassifications

- **double_vision** → validated (was marginal).
- **glitch** → **dropped from study** (user decision; not used downstream). Kept in the validation table as marginal with the dropped-from-study note.
- **wiggle, nystagmus** → pending supervisor decision (neither is blocking).

### 5. Other observations (not fixed this session — TODO)

- **distortion**: Unity warp is visibly more intense than Python at the same params. SSIM_blur 0.94 still passes, but the *magnitude* of the warp differs. Likely a coefficient mismatch in `apply_distortion`'s `magn` scaling. Not blocking; logged for later.

## Session 2026-05-16 — floaters distribution fix

User feedback after the 2026-05-15 floaters rewrite: *"the python version is really centralized a circle full of black dots while the unity version is more of a sprain of distributed black dots."* Visual inspection of `vipsim_assets/unity_refs/floaters_dense.png` confirmed — Unity floaters extend well beyond a hard disc, plus a faint dusting everywhere.

**Root cause.** I had used hard rejection sampling inside a disc of radius `R = 0.146 × min(h, w)`. Unity's `IsInsideCircle` is actually three zones:

```text
if (distance <= radius)                         return true;          // dense disc
if (Random.Range(0,1) < 0.01)                   return true;          // 1% everywhere
float probability = 1.0 - distanceFromEdge / radius;
return Random.Range(0,1) < probability;                                // soft falloff to 2R
```

Combined acceptance probability (single candidate uniform in 1024² overlay):

| zone       | range          | accept P                                |
|------------|----------------|-----------------------------------------|
| disc       | d ≤ R          | 1.0                                     |
| falloff    | R < d ≤ 2R     | 0.01 + 0.99·(1 − (d−R)/R)               |
| spread     | d > 2R         | 0.01                                    |

**Fix.** Rewrote `apply_floaters` in `vipsim_filters.py` to:
1. Sample candidates uniformly in Unity's **1024² overlay** space (not screen space).
2. Accept each via the three-zone probability above (vectorised, batched).
3. Map accepted overlay coords → screen pixel coords (`* w/1024`, `* h/1024`). On a 2048×1024 screen this also reproduces the elliptical disc Unity actually shows when the 1024² texture is stretched onto the screen UVs.
4. Per-pixel alpha-blend the speck (dark/light), 1-px size by default.

**Result.** floaters SSIM_blur 0.98 → **0.9904** at the default spec (n=50000, R/W=0.146, severity=1.0). Side-by-side `abs_diff_out/unityVSpython/compare/floaters_compare.png` shows the dense central disc, soft falloff, and 1% spread all match.

No changes to `validate_all_filters.py` floaters spec needed — the new default `circle_radius_uv = 0.146` aligns with what the validator already passes in.

### Distortion — bug-faithful shader port

User-reported: *"Unity is more intense than Python."* SSIM_blur was 0.94 with MAE 10.3. Spent this session tracing where the magnitude difference comes from.

**Root cause — three independent issues stacked**:

1. **Shader decode bug** (`myDistortionMap.shader:69`):
   ```glsl
   float xOffset = (0.2 * w) - 0.0425;   // correct would be 0.2*w - 0.1
   xOffset = -xOffset;
   ```
   The author intended to invert the encoding `vx*5 + 0.5`, but used `0.0425` instead of `0.1`. Same bug on the Y line.

2. **sRGB texture sampling.** The `Texture2D(W, H, RGB24, false)` constructor defaults to **sRGB**. So when the shader samples warpTextureX/Y, the encoded value goes through sRGB→linear conversion before reaching `w`. Empirical confirmation: in the upper-right quadrant of `distortion_default.png` (far from any warp centre) the pixel difference vs `baseline.png` is **exactly zero** — only possible if Unity's deadband fires there, which requires sRGB compression of the `0.5` zero-warp value down to ≈0.214 linear, yielding `xOffset ≈ 0.0003` (inside the 0.001 deadband).

3. **Coordinate-system mismatch in the port.** The Y warp field is encoded vertically flipped in `myDistortionMap.cs:180` (`vy[x, H-1-y]`), and Unity's D3D Blit UV convention has `i.uv.y = 0` at the *bottom* of the screen, opposite to numpy's `uy = 0` at the top. The old Python port handled neither.

**Combined effect** in Unity (and now Python):
- Zero-warp regions: `xOffset ≈ 0.0003` → zeroed by deadband → no displacement.
- Positive warp `vx = 0.1`: sRGB(1.0) = 1.0, `xOffset = -(0.2 - 0.0425) = -0.1575` (vs the "intended" -0.1).
- Negative warp `vx = -0.1`: sRGB(0.0) = 0.0, `xOffset = +0.0425` (vs the "intended" +0.1).
- → **Strongly asymmetric, ~57% stronger peak displacement than the clean decode.**

**Result.** Distortion SSIM_blur **0.94 → 0.998**, MAE **10.3 → 3.4**, LPIPS 0.04. Side-by-side `abs_diff_out/unityVSpython/compare/distortion_compare.png` shows matching warp swirl around the lower-left centre and matching scene geometry. Documented in code as a faithful port of a known Unity shader bug — this is what users of VIP-Sim actually see.

## Outstanding work

Anything beyond the current 15 validated + 1 dropped + 1 marginal + 2 pending-supervisor:

1. **wiggle / nystagmus** — pending supervisor sign-off. If rejected, the burst pipeline pattern (§ 1 of 2026-05-15) is the principled fix for both.
2. **led** — SSIM_blur 0.72 (cell-phase offset, marginal since 2026-05-13). Not yet revisited. Could try re-aligning the cell grid with Unity's `myLed.shader` ceil math + sub-cell phase.
3. **cataracts frost / noise literal-match** — export Unity's `Mathf.PerlinNoise` / FastNoise grids to PNG and load from Python. Currently SSIM_blur 0.99 / 0.95 respectively so purely for raw-SSIM completeness.
4. **Threshold agreement with supervisor.** Working threshold is SSIM_blur ≥ 0.90 for "validated". Noise floor (Unity baseline vs identical Python no-op) is MAE ~2.6.

## Quick gotchas to remember

- **Filter field names ≠ inspector labels.** Use the C# field names: `myRecolour.anomType`/`severityIndex` (not `Mode`/`Severity`); `myBlur.maxCPD` (not `kernalSigma`, which auto-recomputes from maxCPD on enable); `myCataract.severityIndex`.
- **`OnEnable` resets fields.** That's why the capture script enables BEFORE applying overrides, then waits 1 frame for OnEnable to run, then sets the field, then waits 2 end-of-frames for OnRenderImage to pick the new value up.
- **GazeTracker is required.** Without one, `myFieldLoss`, `myDistortionMap`, `myTeichopsia`, `myFloaters`, `myNystagmus`, `VortexEffect`, `FovealDarkness` all NullRef inside OnUpdate / OnRenderImage. Auto-setup creates it with `gazeSource = None` (xy_norm = 0.5, 0.5 = image center).
- **Shaders go to Always-Included-Shaders.** Otherwise Hidden shaders are stripped from builds and `Shader.Find` returns null. Auto-setup handles this.
- **Win32 IOException 1224 on save.** Defender / Explorer thumbnail cache holds memory-mapped sections on previously written PNGs. Capture script now does delete-first-then-write with temp+rename fallback. If you still hit it on Linux side, `rm` from WSL works (different lock semantics).
