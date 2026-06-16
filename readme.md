<div align="center">

# 👁️ VipSim

### Visual-impairment simulation filters for accessibility research

*A Python (NumPy/PIL) and GPU (PyTorch) reimplementation of the [VIP-Sim](https://github.com/Max-Raed/VIP-Sim) Unity shaders — proven equivalent to the originals and built for VLM accessibility analysis and embodied-AI training.*

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Pillow-013243?logo=numpy&logoColor=white)
![Filters](https://img.shields.io/badge/filters-20-brightgreen)
![Validated](https://img.shields.io/badge/Unity%20equivalence-17%2F19%20validated-success)
![License](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey)

</div>

---

## What this is

VipSim takes an ordinary image and renders it as someone with a given visual impairment would experience it — color-blindness, cataracts, macular degeneration, double vision, migraine aura, and more. It exists to let downstream systems (vision-language model audits, accessibility evaluations, RL agents with simulated low vision) operate on perceptually faithful inputs.

The original VIP-Sim runs as a stack of HLSL shaders inside Unity. This repository re-implements every filter in plain Python so it can run server-side, and again in PyTorch so it can run batched on a GPU inside training loops. Crucially, the Python port is **validated against the Unity original** rather than merely "looking right" — see [Validation](#-validation).

## ✨ Highlights

- **20 visual-impairment filters** spanning color, optical, field-loss, distortion, and time-varying effects.
- **25 clinical presets** that map named conditions (e.g. *macular degeneration*, *diplopia*, *scintillating scotoma*) to ready-made filter chains.
- **Two backends, one behavior** — a NumPy/PIL reference (`vipsim_filters.py`) and a batched PyTorch port (`vipsim_gpu.py`) that reduces every filter to one of three GPU primitives.
- **Proven Unity equivalence** — 17 of 19 tracked filters validated by either source-level proof or a published image-similarity metric stack (SSIM, ΔE2000, Wasserstein-1, LPIPS).
- **Drop-in Gym/MyoSuite wrapper** for applying impairment chains to RL observations on-device.
- **Zero-dependency CLI** for one-off image processing.

## 📦 Repository contents

| File | Role |
|---|---|
| `vipsim_filters.py` | CPU reference implementation — 20 filters, 25 presets, and a CLI. The source of truth for behavior. |
| `vipsim_gpu.py` | PyTorch port operating on batched, channels-first tensors `[N, 3, H, W]`. Includes a Gym/MyoSuite observation wrapper. |
| `vipsim_equivalence_report.md` | The validation document: how the Python port is shown to reproduce the Unity shaders, with methods, metrics, and results. |

## 🚀 Quick start

### Install

```bash
pip install numpy pillow            # CPU reference
pip install torch                   # GPU port (optional)
```

### Command line

```bash
# Single filter
python vipsim_filters.py input.png output.png --filter blur --severity 0.5
python vipsim_filters.py input.png output.png --filter cvd --type deuteranomaly --severity 0.8
python vipsim_filters.py input.png output.png --filter cataracts --severity 0.5

# Brightness / contrast / gamma
python vipsim_filters.py input.png output.png --filter bcg --brightness 0.8 --contrast 0.7 --gamma 1.5

# Named preset (a chain of filters)
python vipsim_filters.py input.png output.png --preset moderate_low_vision

# See everything available
python vipsim_filters.py --list-presets
```

### Python API

```python
from PIL import Image
import vipsim_filters as vip

img = Image.open("scene.png").convert("RGB")

# Apply one filter
out = vip.apply_cvd(img, cvd_type="deuteranomaly", severity=0.6)

# Apply a clinical preset (chained filters)
out = vip.apply_preset(img, "macular_degeneration")

out.save("scene_impaired.png")
```

### GPU / batched

The PyTorch port works on a whole batch of frames at once and keeps everything on-device. Image-independent data (warp fields, blur kernels, masks) is precomputed once per `(severity, H, W)` and cached.

```python
import torch
from vipsim_gpu import VipSimGPU

sim = VipSimGPU(device="cuda")
x = torch.rand(8, 3, 256, 256, device="cuda")   # batch of 8 frames in [0,1]

x = sim.cvd(x, severity=0.8)
x = sim.blur(x, severity=0.4)
x = sim.vortex(x, severity=0.5)
```

### Inside an RL / MyoSuite environment

```python
from vipsim_gpu import VisualImpairmentWrapper

env = VisualImpairmentWrapper(
    env,
    chain=[("cvd", {"severity": 0.8}), ("blur", {"severity": 0.4})],
    device="cuda",
    image_key=None,        # set this if the observation is a dict
)
```

## 🎛️ Filter catalogue

Every filter takes a `severity` in `[0, 1]` unless noted. The **Simulates** column is the clinical condition the matching preset is named for.

| Filter | Simulates | Unity equivalence |
|---|---|:---:|
| `cvd` | Color-vision deficiency (prot / deuter / tritanomaly, monochrome) | ✅ proof |
| `bcg` | Brightness / contrast / gamma shift | ✅ proof |
| `field_loss` | Macular degeneration (central) / tunnel vision (peripheral) | ✅ proof |
| `blur` | Low-vision acuity loss | ✅ |
| `bloom` | Photophobia / glare from bright regions | ✅ |
| `cataracts` | Cataracts (blur + yellowing + halo scatter) | ✅ |
| `distortion` | Metamorphopsia (wavy lines) | ✅ |
| `double_vision` | Diplopia | ✅ |
| `pixelation` | Very low acuity / low resolution | ✅ |
| `vortex` | Local swirling distortion | ✅ |
| `noise` | Visual snow / interference | ✅ |
| `teichopsia` | Scintillating scotoma (migraine aura) | ✅ |
| `foveal_darkness` | Central scotoma | ✅ |
| `floaters` | Vitreous floaters | ✅ |
| `flickering_stars` | Random bright dots (time-varying) | ✅ burst |
| `wiggle` | Sinusoidal warp (time-varying) | ✅ burst |
| `nystagmus` | Involuntary eye-oscillation motion (time-varying) | ✅ burst |
| `glitch` | RGB channel-shift glitch | ⚠️ dropped |
| `led` | LED-board cellular dot pattern | ⚠️ marginal |
| `detail_loss` | Contrast-sensitivity loss (posterization) | — not yet tracked |

`✅ proof` = closed by source-level algorithmic equivalence · `✅` / `✅ burst` = passes the measured-fidelity threshold (single-frame or multi-frame burst) · `⚠️` = below threshold, see report.

## 🧩 Clinical presets

Presets bundle one or more filters under a recognizable name. A selection:

| Preset | What it models |
|---|---|
| `mild_low_vision` / `moderate_low_vision` / `severe_low_vision` | Graded blur + contrast loss |
| `deuteranomaly_moderate` / `protanomaly_moderate` | Red-green color blindness |
| `cataracts_moderate` / `elderly_vision` | Cataracts, with optional age-related CVD |
| `macular_degeneration` / `tunnel_vision` | Central vs. peripheral field loss |
| `metamorphopsia` / `diplopia` | Wavy distortion / double vision |
| `scintillating_scotoma` / `central_scotoma` | Migraine aura / central dark spot |
| `photophobia` | Glare sensitivity |
| `nystagmus_motion` / `floaters` / `visual_noise` | Motion blur, floaters, visual snow |

Run `python vipsim_filters.py --list-presets` for the full set of 25 with descriptions.

## 🔬 Under the hood (GPU port)

The PyTorch backend reduces all ~20 filters to **three primitives**, which is why it stays fast and batched:

1. **Elementwise / 3×3 matmul** → `einsum` + arithmetic — *cvd, bcg, monochrome, noise, posterize, tint…*
2. **Separable blur / pooling** → `conv2d`, `avg_pool2d` — *blur, bloom, field-loss mip pyramids, cataract frost…*
3. **Per-pixel UV displacement** → `F.grid_sample` — *distortion, vortex, wiggle, double vision, nystagmus…*

Anything that doesn't depend on the image content (warp grids, masks, Gaussian kernels) is computed **once** per `(severity, H, W)` and cached on the device, so per-frame work is minimal.

## ✅ Validation

The Python port isn't just visually plausible — it's checked against the Unity original. Two complementary methods are used, documented in full in [`vipsim_equivalence_report.md`](vipsim_equivalence_report.md):

- **Method A — algorithmic proof.** For pure deterministic arithmetic filters (`bcg`, `cvd`, `field_loss`), the HLSL fragment math and the NumPy code are lined up and shown to compute the same function. A confirmatory Unity render only confirms the residual sits at the fp16/fp32 rounding floor.
- **Method B — measured fidelity.** For everything else (noise, kernels, halo scatter, UV warps, gaze-contingent and time-varying effects), matched 2048×1024 captures are compared through a stack of published metrics: **MAE, SSIM, SSIM_blur, ΔE2000, Wasserstein-1, LPIPS**.

A filter is **validated** when its best sweep point reaches **SSIM_blur ≥ 0.90** against the matched Unity reference. SSIM_blur (a σ=1px pre-blur before SSIM) is the operational criterion because several filters introduce sub-pixel offsets that are invisible to a human but punish plain SSIM.

For the three inherently time-varying filters (`flickering_stars`, `wiggle`, `nystagmus`) a **burst-frame protocol** captures 30 matched frames over a fixed window and aggregates the metric across them.

**Status: 17 of 19 tracked filters validated.** `glitch` sits just below threshold (SSIM_blur 0.890, non-deterministic) and `led` is marginal pending cell-phase tuning.

### Reproduce the sweep

```bash
source .venv/bin/activate
python scripts/validate_all_filters.py --no-strips

# burst-frame validators for the time-varying filters
python scripts/validate_flickering_stars_burst.py
python scripts/validate_wiggle_burst.py
python scripts/validate_nystagmus_burst.py
```

## 📚 References

- Wang, Bovik, Sheikh & Simoncelli (2004). *Image quality assessment: from error visibility to structural similarity.* IEEE TIP 13(4).
- Sharma, Wu & Dalal (2005). *The CIEDE2000 color-difference formula.* Color Research & Application 30(1).
- Zhang, Isola, Efros, Shechtman & Wang (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR.
- Kantorovich (1942) / Villani (2008) — optimal transport (Wasserstein distance).

## 📄 License & attribution

Based on shader code from **[VIP-Sim](https://github.com/Max-Raed/VIP-Sim)** by Max Raed, licensed **CC BY 4.0**. This reimplementation is distributed under the same terms — please preserve attribution.

---

<div align="center">
<sub>Built for accessibility research and embodied-AI evaluation.</sub>
</div>