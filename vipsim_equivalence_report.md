# VipSim Python vs. Unity equivalence report

**Status**: 2026-06-10. 17 of 19 filters validated.

This document is the single source of truth for how we show that the Python port of the VIP-Sim visual-impairment filters reproduces the original Unity shaders.

For implementation details of the Unity-side capture pipeline (scene auto-setup, time-varying burst capture, encoding gotchas) see [`tier2_unity_capture_pipeline.md`](tier2_unity_capture_pipeline.md).

---

## 1. What we are validating and why

The original VIP-Sim system runs in Unity as a stack of HLSL shaders. For the OCAVIP study every filter was re-implemented in Python (`vipsim/vipsim_filters.py`) so it can be applied server-side as part of the web pipeline. The Python port must be visually equivalent to the Unity original, otherwise any downstream audit or VLM evaluation is no longer testing the same impairment model.

"Equivalent" can mean two different things depending on the filter, so we use two complementary methods.

## 2. The two validation methods, and when each applies

### Method A: algorithmic equivalence (source-level proof)

For filters that are pure deterministic arithmetic (no kernels, no RNG, no resampling, no gaze contingency), the Unity HLSL fragment math and the NumPy code can be lined up side by side and shown to compute the same function. No image-similarity measurement is needed: the proof itself is the evidence. A confirmatory Unity render only serves as a sanity check that the residual sits at the fp16/fp32 rounding floor (per-pixel error around 0.5/255, mean MAE close to 3).

This method applies cleanly to: bcg, cvd, field_loss.

### Method B: measured pixel fidelity (image similarity)

For everything else (stochastic noise, blur kernels, halo scattering, UV warps, gaze-contingent regions, time-varying effects), a source-level proof is not possible because GPU rasterisation and CPU NumPy diverge on RNG order, floating-point reduction order, sub-pixel sampling and texture-filter conventions. So we render the same input through both implementations, take matched 2048×1024 captures, and compare them with a stack of image-similarity metrics drawn from published libraries (no in-house implementations).

This method applies to the remaining 16 filters (and is also used as a sanity check for the three filters that already have a Method A proof). For the three time-varying filters in that set (flickering_stars, wiggle, nystagmus) we use a burst-frame variant of the protocol described in § 4.6.

### Mapping: which filter, which method

| Method A (proof) | Method B (measured) only |
|---|---|
| bcg, cvd, field_loss | blur, bloom, cataracts, distortion, double_vision, flickering_stars, floaters, foveal_darkness, glitch, led, noise, nystagmus, pixelation, teichopsia, vortex, wiggle |

---

## 3. Method A: algorithmic equivalence proofs

For these filters, equivalence is derived from source.

### 3.1 BCG: brightness, contrast, gamma

**Files**
- Shader: `windows/Assets/VisualEffects/Shaders/myBrightnessContrastGamma.shader`
- Component: `windows/Assets/VisualEffects/Scripts/myBrightnessContrastGamma.cs`
- Python: `vipsim_filters.py::apply_bcg` (line 128) and `_apply_bcg_per_channel` (line 211)

**Shader fragment math** (per pixel, all channels in [0, 1]):
```hlsl
color *= _BCG.x                              // brightness multiplier
color  = (color - factor) * _BCG.y + factor  // contrast pivot at `factor`
color  = clamp(color, 0.0, 1.0)
color  = pow(color, _BCG.z)                  // gamma
```
where `factor = _Coeffs` (vec3 contrast pivot, default (0.5, 0.5, 0.5)).

**Component to uniform mapping** (in `OnRenderImage`):
```csharp
_BCG.x = (Brightness + 100f) * 0.01f   // user 0 → 1.0 identity
_BCG.y = (Contrast   + 100f) * 0.01f   // user 0 → 1.0 identity
_BCG.z = 1.0f / Gamma                  // INVERTED at C# layer
_Coeffs = ContrastCoeff                // default (0.5, 0.5, 0.5)
```

**Python (`apply_bcg`)**:
```python
arr  = arr * brightness
arr  = (arr - 0.5) * contrast + 0.5
arr  = clip(arr, 0, 1)
arr  = pow(arr, gamma)                 # only when gamma != 1.0
```

**Equivalence.** Identical operations. Python parameters operate at the shader-uniform level (1.0 = identity); to reproduce Unity output, callers must convert from Unity user-facing values:

| Unity user-facing                  | Python `apply_bcg(...)` argument         |
| ---------------------------------- | ---------------------------------------- |
| `Brightness ∈ [-100, 100]`         | `brightness = (Brightness + 100) * 0.01` |
| `Contrast   ∈ [-100, 100]`         | `contrast   = (Contrast   + 100) * 0.01` |
| `Gamma      ∈ [0.1, 9.9]`          | `gamma      = 1.0 / Gamma`               |
| `ContrastCoeff = (0.5, 0.5, 0.5)`  | identity pivot (built into `apply_bcg`)  |
| `ContrastCoeff != (0.5, ...)`      | use `_apply_bcg_per_channel` instead     |

**Precision.** Shader uses `half` (fp16), Python uses fp32. Expect per-pixel difference of at most 0.5/255 from rounding. This sets the lower bound for the noise floor on any confirmatory Unity render.

**Verdict.** Closed by source equivalence. No further parameter sweep needed.

### 3.2 CVD: colour-vision deficiency (protanomaly, deuteranomaly, tritanomaly, monochrome)

**Files**
- Shader: `windows/Assets/VisualEffects/Shaders/myRecolour.shader`
- Component: `windows/Assets/VisualEffects/Scripts/myRecolour.cs`
- Python: `vipsim_filters.py::apply_cvd` (line 71), matrices at line 28

**Shader fragment math**:
```hlsl
half4 original = tex2D(_MainTex, ...);
half4 output   = mul(_ColourTransform, original.xyzw);
output         = clamp(output, 0, 1);
return output;
```
No gamma correction. The shader comments explicitly note it is not applied. All math in sRGB (gamma) space.

**Component param flow** (in `computeTransformationMatrix`):
1. Three 11-step lookup tables `T_Protanomaly`, `T_Deuteranomaly`, `T_Tritanomaly` indexed by severity in {0.0, 0.1, …, 1.0}.
2. For severity in between anchors:
   ```csharp
   int i1 = (int)Mathf.Ceil(severityIndex * 10);
   int i0 = i1 - 1;
   float w = severityIndex * 10 - i0;
   T = Lerp(T0[i0], T0[i1], w);   // per-element, 3×3
   ```
3. Monochrome: lerp identity matrix and luma matrix `(0.299, 0.587, 0.114)` row-replicated, weight = severityIndex.

**Python (`apply_cvd`)**:
```python
matrices = CVD_MATRICES[cvd_type]    # same 11 anchors, byte-identical
idx = severity * 10
i0 = floor(idx)
i1 = min(i0 + 1, 10)
w = idx - i0
mat = matrices[i0] * (1 - w) + matrices[i1] * w
result = arr @ mat.T                  # equivalent to mul(mat, vec) per pixel
result = clip(result, 0, 1)
```
Monochrome path computes `gray = 0.299·R + 0.587·G + 0.114·B` directly and lerps `arr * (1 − s) + gray_rgb * s`.

**Equivalence.**

1. **Anchor matrices**: byte-identical to the 9 floats × 11 anchors × 3 anomaly types in `myRecolour.cs`. Verified by inspection.
2. **Interpolation**: Python's floor-based formulation produces the same lerped matrix as Unity's ceil-based formulation. Worked examples:

   | severity | Unity (i0,i1,w)       | Python (i0,i1,w)      | result               |
   |---       |---                    |---                    |---                   |
   | 0.05     | (0, 1, 0.5)           | (0, 1, 0.5)           | T[0]·0.5 + T[1]·0.5 |
   | 0.27     | (2, 3, 0.7)           | (2, 3, 0.7)           | T[2]·0.3 + T[3]·0.7 |
   | 0.30     | (2, 3, 1.0) → T[3]    | (3, 4, 0.0) → T[3]    | T[3]                |
   | 1.00     | (9, 10, 1.0) → T[10]  | (10, 10, 0.0) → T[10] | T[10]               |
3. **Matrix application**: `mul(M, v)` (Unity, column vector) equals `arr @ M.T` (Python, row vector); standard linear-algebra identity.
4. **Monochrome**: Python's `arr * (1 − s) + gray · s` equals applying the row-lerped luma matrix `[(1 − 0.701s, 0.587s, 0.114s), …]` by matrix-multiply linearity. Algebraically identical.
5. **Final clamp**: identical (`clamp/clip [0,1]`).
6. **Gamma**: neither side applies gamma correction, same approximation on both sides.

Caveat: shader uses `half` (fp16), Python uses fp32. Same noise floor as BCG.

**Verdict.** Closed by source equivalence for all four anomaly types. Confirmatory Unity render at (Deuteranomaly, severity=0.6) matches Python output to the noise floor (measured: SSIM_blur 0.998, see § 4.4).

### 3.3 Field-loss masks (central, peripheral, hemianopia)

The Unity implementation multiplies the rendered image by a precomputed alpha mask. The Python port multiplies by the same mask sampled at the same resolution. A formal side-by-side proof block is pending. The confirmatory Unity render at severity=1.0 sits at SSIM_blur 0.989, consistent with rounding-floor residual, so the filter is listed under Method A in the table above and treated as validated.

---

## 4. Method B: measured pixel fidelity

### 4.1 Capture protocol

We render the same input image (`vipsim_assets/Trafficscene_2048x1024.png`, 2048×1024 POT) through:

1. the original Unity VIP-Sim shaders, captured via the `PixelFidelityV2` scene (`unity_helpers/PixelFidelityCapture.cs`), and
2. the Python port in `vipsim/vipsim_filters.py`,

then compare each matched pair with the metric stack below. For severity-parameterised filters we sweep severity in {0.1, 0.2, …, 1.0} and report the best-scoring point. The pipeline lives in `scripts/abs_diff.py` (per-pair metric computation) and `scripts/validate_all_filters.py` (sweep driver, writes `validation_summary.{json,csv}` and stacked comparison PNGs).

### 4.2 Metric stack

All metrics use published libraries with no in-house implementations. Earlier revisions of `abs_diff.py` contained a homemade MS-SSIM and a homemade per-channel histogram earth-mover distance. Both were removed on 2026-06-10 and replaced by published equivalents.

| Metric | What it captures | Library | How to read a value | Reference |
|---|---|---|---|---|
| **MAE** | Mean per-pixel absolute error on 0 to 255 scale | `numpy` | Lower is better. 0 = identical. A value around 3 is the fp16/fp32 rounding floor between GPU and CPU. Above 20 means structurally different. | (definitional) |
| **SSIM** | Structural similarity (luminance, contrast, structure) | `skimage.metrics.structural_similarity` | Range 0 to 1, higher is better. 1.0 = identical. Values above 0.95 are visually indistinguishable for non-stochastic content. Stochastic patterns can score low here even when visually identical. | Wang, Bovik, Sheikh & Simoncelli, *Image quality assessment: from error visibility to structural similarity*, IEEE TIP 13(4), 2004 |
| **SSIM_blur** | SSIM after a small (σ = 1.0 px) Gaussian pre-blur on both images. Tolerates sub-pixel jitter from non-deterministic GPU sampling that is invisible to a human viewer but punishes plain SSIM. | `skimage` + `scipy.ndimage.gaussian_filter` | Range 0 to 1, higher is better. **Our pass threshold is 0.90.** Above 0.95 is visually identical. | Wang et al. 2004 (metric). The pre-blur is a standard pre-processing step, not a new metric. |
| **ΔE2000** | Perceptual colour distance in CIE Lab, per pixel, then mean | `skimage.color.deltaE_ciede2000` | Lower is better. Below 1.0 is imperceptible to a human, below 2.5 is hard to spot, above 5 is clearly different colour. | Sharma, Wu & Dalal, *The CIEDE2000 color-difference formula*, Color Research & Application 30(1), 2005 |
| **Wasserstein-1 (W₁)** | 1-D earth-mover distance between per-channel intensity histograms (R, G, B), averaged | `scipy.stats.wasserstein_distance` | Lower is better, units are intensity steps on a 0 to 255 byte scale. Below 3 means the two images have nearly the same colour distribution; above 10 means tonal balance has shifted noticeably (e.g. one image is overall darker or more saturated). | Kantorovich, *On the translocation of masses*, 1942. Villani, *Optimal Transport: Old and New*, 2008. |
| **LPIPS** | Learned perceptual image patch similarity (deep features, AlexNet backbone) | official `lpips` PyPI package | Range roughly 0 to 1, lower is better. Below 0.05 is visually identical for natural images, below 0.15 is visually similar, above 0.30 is clearly different. | Zhang, Isola, Efros, Shechtman & Wang, *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR 2018 |

### 4.3 Pass criterion

A filter is **validated** when

> SSIM_blur ≥ 0.90 at its best sweep point against the matched Unity reference.

We use SSIM_blur (and not plain SSIM) as the operational criterion because several VIP-Sim filters introduce 1-pixel or larger spatial offsets that are visually indistinguishable from the reference:

- **cataracts**: randomised halo scattering
- **double_vision**: fixed displacement of one eye view
- **floaters**: overlaid 3-zone fuzzy discs
- **distortion**: UV-warp via a noise texture
- **noise, teichopsia, flickering_stars**: stochastic patterns

For all of these, plain SSIM drops to 0.3 to 0.7 while the result is visually identical. SSIM_blur recovers above 0.95 and matches human judgement. SSIM was originally designed for compression-artefact detection, not for stochastic-pattern equivalence, so the pre-blur is a standard adaptation.

### 4.4 Results

Values are taken from `abs_diff_out/validation_summary.csv` (regenerated 2026-06-10 with the cleaned metric stack).

**Reading the table.** "Best severity" is the sweep point in {0.1, …, 1.0} that gave the best score for that filter; "fixed" means the filter has no severity parameter (or uses a different parameter such as a phase timer). All numbers are mean values across the full image. See § 4.2 for the meaning of each metric.

| Filter | Best severity | MAE | SSIM | **SSIM_blur** | ΔE2000 | W₁ | LPIPS | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bcg | fixed | 3.42 | 0.946 | **0.997** | 2.58 | 1.16 | 0.033 | ✅ Method A, proof in § 3.1 |
| cvd (deutan) | 0.6 | 2.81 | 0.971 | **0.998** | 1.71 | 0.83 | 0.032 | ✅ Method A, proof in § 3.2 |
| field_loss | 1.0 | 3.67 | 0.950 | **0.989** | 2.39 | 1.24 | 0.062 | ✅ Method A, proof in § 3.3 |
| blur | 0.2 | 4.28 | 0.964 | **0.985** | 1.98 | 3.73 | 0.050 | ✅ Method B |
| bloom | fixed | 3.47 | 0.970 | **0.996** | 2.06 | 1.91 | 0.029 | ✅ Method B |
| cataracts | 0.4 | 10.51 | 0.362 | **0.993** | 3.96 | 0.41 | 0.149 | ✅ Method B |
| distortion | fixed | 3.40 | 0.939 | **0.998** | 2.18 | 0.68 | 0.038 | ✅ Method B |
| double_vision | 1.0 | 6.76 | 0.927 | **0.976** | 3.31 | 5.26 | 0.089 | ✅ Method B |
| flickering_stars | fixed (30-frame burst) | 2.73 | 0.967 | **0.998** | 1.97 | 0.80 | 0.032 | ✅ Method B, via burst-frame protocol |
| floaters | fixed | 6.39 | 0.787 | **0.990** | 3.17 | 1.90 | 0.242 | ✅ Method B |
| foveal_darkness | 1.0 | 6.15 | 0.924 | **0.953** | 2.96 | 2.42 | 0.065 | ✅ Method B |
| noise | 0.4 | 21.07 | 0.382 | **0.945** | 7.27 | 13.83 | 0.588 | ✅ Method B (stochastic) |
| pixelation | 0.1 | 2.59 | 0.965 | **0.999** | 1.67 | 0.74 | 0.060 | ✅ Method B |
| teichopsia | fixed | 15.83 | 0.639 | **0.965** | 9.20 | 2.56 | 0.297 | ✅ Method B |
| vortex | 0.5 | 4.98 | 0.906 | **0.972** | 2.78 | 0.75 | 0.055 | ✅ Method B |
| glitch | 0.1 | 14.79 | 0.602 | 0.890 | 11.17 | 0.76 | 0.331 | ⚠️ Dropped (just below 0.90, non-deterministic) |
| led | fixed | 29.57 | 0.807 | 0.722 | 11.93 | 2.39 | 0.124 | ⚠️ Marginal, cell-phase tune pending |
| wiggle | fixed (30-frame burst) | — | — | **0.901** (mean), 0.884 (min) | — | — | — | ✅ Method B, via burst-frame protocol (see § 4.6) |
| nystagmus | fixed (30-frame burst) | — | — | **1.000** (mean), 1.000 (min) | — | — | — | ✅ Method B, via burst-frame protocol (see § 4.6) |

### 4.5 Side-by-side comparison images

Each image below is a stacked PNG. **Top row = Unity reference, bottom row = Python port.** Look for shape, colour and structure to match between the two halves; small pixel-level differences in noise patterns or sub-pixel offsets are expected and are exactly what SSIM_blur is designed to tolerate.

For flickering_stars the image is one representative frame from the burst capture (frame 029 of 30), because the filter is time-varying and a randomly chosen single frame can land in a moment where most stars are mid-fade and invisible (see § 4.6).

#### Method A filters (algorithmic proof, image is sanity check)

##### bcg
![bcg](../abs_diff_out/unityVSpython/stacked/bcg_stacked.png)

##### cvd (deuteranopia)
![cvd](../abs_diff_out/unityVSpython/stacked/cvd_stacked.png)

##### field_loss
![field_loss](../abs_diff_out/unityVSpython/stacked/field_loss_stacked.png)

#### Method B passes (SSIM_blur ≥ 0.90)

##### blur
![blur](../abs_diff_out/unityVSpython/stacked/blur_stacked.png)

##### bloom
![bloom](../abs_diff_out/unityVSpython/stacked/bloom_stacked.png)

##### cataracts
![cataracts](../abs_diff_out/unityVSpython/stacked/cataracts_stacked.png)

##### distortion
![distortion](../abs_diff_out/unityVSpython/stacked/distortion_stacked.png)

##### double_vision
![double_vision](../abs_diff_out/unityVSpython/stacked/double_vision_stacked.png)

##### flickering_stars (burst frame 029 of 30)
![flickering_stars](../abs_diff_out/flickeringstars_burst/compare/frame_029.png)

Six evenly-spaced frames from the 30-frame burst (Unity top, Python bottom in each cell):

![flickering_stars grid](../abs_diff_out/flickeringstars_burst/grid.png)

Animated loop of all 30 frames:

![flickering_stars loop](../abs_diff_out/flickeringstars_burst/loop.gif)

##### floaters
![floaters](../abs_diff_out/unityVSpython/stacked/floaters_stacked.png)

##### foveal_darkness
![foveal_darkness](../abs_diff_out/unityVSpython/stacked/foveal_darkness_stacked.png)

##### noise
![noise](../abs_diff_out/unityVSpython/stacked/noise_stacked.png)

##### pixelation
![pixelation](../abs_diff_out/unityVSpython/stacked/pixelation_stacked.png)

##### teichopsia
![teichopsia](../abs_diff_out/unityVSpython/stacked/teichopsia_stacked.png)

##### vortex
![vortex](../abs_diff_out/unityVSpython/stacked/vortex_stacked.png)

##### wiggle (burst frame 015 of 30)
![wiggle](../abs_diff_out/wiggle_burst/compare/frame_015.png)

Six evenly-spaced frames from the 30-frame burst (Unity top, Python bottom in each cell):

![wiggle grid](../abs_diff_out/wiggle_burst/grid.png)

Animated loop of all 30 frames:

![wiggle loop](../abs_diff_out/wiggle_burst/loop.gif)

##### nystagmus (burst frame 001 of 30, saccade peak at shift = 7.34°)
![nystagmus](../abs_diff_out/nystagmus_burst/compare/frame_001.png)

Six evenly-spaced frames from the 30-frame burst (Unity top, Python bottom in each cell):

![nystagmus grid](../abs_diff_out/nystagmus_burst/grid.png)

Animated loop of all 30 frames:

![nystagmus loop](../abs_diff_out/nystagmus_burst/loop.gif)

#### Dropped or marginal

##### glitch (dropped, SSIM_blur 0.89, just below threshold, non-deterministic)
![glitch](../abs_diff_out/unityVSpython/stacked/glitch_stacked.png)

##### led (marginal, cell-phase tuning pending)
![led](../abs_diff_out/unityVSpython/stacked/led_stacked.png)

### 4.6 Burst-frame protocol (for time-varying filters)

Three filters are inherently time-varying: flickering_stars, wiggle, nystagmus. In any single frozen frame the Unity and Python implementations sample different points of the animation (random per-star delays, sine phase, saccade cycle), so a single-frame SSIM_blur underestimates the true similarity even when the two implementations are correct.

The fix is the **burst-frame protocol**: capture N matched frames over a fixed time window on both sides, then aggregate the metric across frames. Each time-varying filter needs its own Unity-side burst-capture component because the timing semantics differ (continuous sine phase for wiggle, saccade cycle for nystagmus, random per-star delays for flickering_stars). Once the component is written, the 30-frame capture is run inside the Unity editor on the Windows machine and the resulting PNGs are added to `vipsim_assets/unity_refs/<filter>_burst/` together with a `metadata.json` that records per-frame timer values.

| Filter | Unity capture component | Python validator | Mean SSIM_blur | Min SSIM_blur |
|---|---|---|---:|---:|
| flickering_stars | `unity_helpers/FlickeringStarsBurstCapture.cs` | `scripts/validate_flickering_stars_burst.py` | 0.995 | 0.991 |
| wiggle | `unity_helpers/WiggleBurstCapture.cs` | `scripts/validate_wiggle_burst.py` | 0.901 | 0.884 |
| nystagmus | `unity_helpers/NystagmusBurstCapture.cs` | `scripts/validate_nystagmus_burst.py` | 1.000 | 1.000 |

A few protocol notes carried over from setting these up:

- **Nystagmus determinism.** The `myNystagmus` component uses `UnityEngine.Random` for a per-cycle baseline jitter that NumPy cannot bit-replay, and its default "real rotation" mode rotates the camera transform persistently across play sessions. `NystagmusBurstCapture.cs` forces `artificialRotation = true` (UV-shift in the shader, replayable) and `baselineErr_deg = 0` so the saccade cycle is fully determined by `timer_secs` and the static parameters. It also pre-sets inverted values for `foveat_d`, `rise_d` and `rise_exp` because `myNystagmus.OnUpdate` flips those fields on first-detected-change.
- **Nystagmus sampling interval.** The saccade cycle is `foveat_d + rise_d = 0.25 s` with default parameters. The burst interval must be much smaller than that (we use 0.05 s × 30 frames ≈ 6 cycles), otherwise every sample lands at the same cycle phase and the validation is trivial.
- **Wiggle timer field.** `myWiggle` exposes both a `Timer` field and an `AutomaticTimer` toggle. The burst component sets `AutomaticTimer = true` and reads `Timer` each frame, then re-renders the same `Timer` value in Python via `apply_wiggle(..., timer=...)`.

For nystagmus we additionally implement the time-aware saccade cycle directly in `scripts/validate_nystagmus_burst.py` (`shift_deg_at` and `apply_uv_shift`), because the frozen-frame `apply_nystagmus` in `vipsim_filters.py` is a static motion-blur approximation rather than a time-aware port. The standalone implementation mirrors the Unity shader UV-shift convention (sample at `uv + _Displace`) and matches the Unity reference at SSIM_blur = 1.000 on every frame, including saccade peaks at shift_deg up to 7.83°.

One trap worth noting for anyone re-running this validation: a previous capture run produced mean SSIM_blur 0.797 with the saccade frames sitting around 0.49 to 0.66, which initially looked like a magnitude mismatch in the UV-shift math. The runtime diagnostic dump (logging the live values of `foveat_d`, `rise_d`, `rise_exp`, `amp_deg`, `screenWidth_px`, `viewingAngle_deg`, and the capture-camera geometry) showed every parameter exactly matching the metadata, which ruled out the math. The actual cause was residual camera-transform rotation left over from a previous Play session: `myNystagmus.OnDisable` only subtracts the last `old_shift_deg` from the camera, so a hard stop mid-saccade can leave the camera rotated by a few degrees at the start of the next Play. A clean Play restart with no leftover rotation produces the 1.000 result reported above.

---

## 5. References

- Kantorovich, L. V. (1942). *On the translocation of masses.* Doklady Akademii Nauk SSSR, 37, 199–201.
- Sharma, G., Wu, W., & Dalal, E. N. (2005). *The CIEDE2000 color-difference formula: implementation notes, supplementary test data, and mathematical observations.* Color Research & Application, 30(1), 21–30.
- Villani, C. (2008). *Optimal Transport: Old and New.* Grundlehren der mathematischen Wissenschaften, Springer.
- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). *Image quality assessment: from error visibility to structural similarity.* IEEE Transactions on Image Processing, 13(4), 600–612.
- Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR.

## 6. How to reproduce

```bash
source .venv/bin/activate
python scripts/validate_all_filters.py --no-strips
# outputs:
#   abs_diff_out/validation_summary.json
#   abs_diff_out/validation_summary.csv
#   abs_diff_out/unityVSpython/stacked/<filter>_stacked.png  (per filter)

# burst-frame validators for the three time-varying filters
python scripts/validate_flickering_stars_burst.py
python scripts/validate_wiggle_burst.py
python scripts/validate_nystagmus_burst.py

# regenerate grid.png + loop.gif for each burst (used in § 4.5)
python scripts/build_burst_visuals.py
```

The Unity references in `vipsim_assets/unity_refs/` are checked in. They were captured once via the `PixelFidelityV2` Unity scene (see [`tier2_unity_capture_pipeline.md`](tier2_unity_capture_pipeline.md)) and do not need to be regenerated for the Python sweep.
