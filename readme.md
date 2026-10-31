<div align="center">

# 👁️ VipSim

### Visual-impairment simulation filters for accessibility research

*A Python (NumPy/PIL) and GPU (PyTorch) reimplementation of the [VIP-Sim](https://github.com/Max-Raed/VIP-Sim) Unity shaders — proven equivalent to the originals and built for VLM accessibility analysis and embodied-AI training.*

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Pillow-013243?logo=numpy&logoColor=white)
![Filters](https://img.shields.io/badge/filters-20-brightgreen)
![Validated](https://img.shields.io/badge/Unity%20equivalence-17%2F19%20validated-success)
![License](https://img.shields.io/badge/license-MIT-blue)
![PyPI](https://img.shields.io/pypi/v/vipsim)

</div>

---

## What this is

VipSim takes an ordinary image and renders it as someone with a given visual impairment would experience it — color-blindness, cataracts, macular degeneration, double vision, migraine aura, and more. It exists to let downstream systems (vision-language model audits, accessibility evaluations, RL agents with simulated low vision) operate on perceptually faithful inputs.

The original VIP-Sim runs as a stack of HLSL shaders inside Unity. This repository re-implements every filter in plain Python so it can run server-side, and again in PyTorch so it can run batched on a GPU inside training loops. Crucially, the Python port is **validated against the Unity original** rather than merely "looking right" — see [Validation](#-validation).

## ✨ Highlights

- **20 visual-impairment filters** spanning color, optical, field-loss, distortion, and time-varying effects.
- **24 clinical presets** that map named conditions (e.g. *macular degeneration*, *diplopia*, *scintillating scotoma*) to ready-made filter chains.
- **Two backends** — a NumPy/PIL reference (`vipsim.filters`) and a batched PyTorch port (`vipsim.gpu`). The GPU port covers every filter, preset and profile, with the CPU's keyword arguments, and reproduces the CPU output bit for bit wherever it has been tested (see [Under the hood](#-under-the-hood-gpu-port)).
- **Checked against Unity** — 17 of 19 tracked filters pass the repo's SSIM_blur criterion against matched Unity captures; read the [caveats](#-validation) before citing this.
- **Drop-in Gym/MyoSuite wrapper** for applying impairment chains to RL observations on-device.
- **`vipsim` command** for one-off image processing, installed with the package.

## 📦 Repository contents

Only the `vipsim/` package goes into the PyPI release; everything else is for development and validation.

| File | Role |
|---|---|
| `vipsim/filters.py` | CPU reference implementation — 20 filters, 24 presets, and the CLI. The source of truth for behavior. |
| `vipsim/gpu.py` | PyTorch port of every filter, operating on batched, channels-first tensors `[N, 3, H, W]` and matching the CPU output. Runs presets and profiles; includes a Gym/MyoSuite observation wrapper. |
| `vipsim/__init__.py`, `vipsim/__main__.py` | Public API (`from vipsim import apply_preset`, …), `__version__`, and `python -m vipsim`. |
| `vipsim/assets/` | The macular-degeneration overlay used by `apply_field_loss` (shipped in the package). |
| `vipsim_filters.py`, `vipsim_gpu.py` | Aliases so older commands (`python vipsim_filters.py …`) keep working; not part of the installed package. |
| `pyproject.toml` | Package metadata and build config (hatchling; distribution name `vipsim`). |
| `.github/workflows/publish.yml` | Builds and publishes to PyPI via Trusted Publishing when a GitHub release is published. |
| `tests/` | Unit tests (CPU filters, CLI, packaging, GPU-vs-CPU parity). |
| `vipsim_equivalence_report.md` | The validation document: how the Python port is shown to reproduce the Unity shaders, with methods, metrics, and results. |
| `validate.sh`, `validation_scripts/` | One-command Unity-equivalence validation and the scripts behind it. |
| `vipsim_assets/`, `abs_diff_out/` | Input scene, Unity reference captures, and validation outputs. |
| `vipsim_profiles_v2/` | The 7 validated VIP-Sim participant profiles as filter stacks (`p1-p7.json`), calibrated against the paper's Figure-6 panels, with side-by-side renders. |
| `validation_scripts/profile_tuner.py` | Interactive browser tuner (localhost:8765): live sliders for every parameter, similarity scoring vs a reference, auto-tune, and a generated `/docs` page. |
| `validation_scripts/render_profiles_v2.py` | Regenerates the paper-vs-ours comparison renders for the profiles. |
| `unity_helpers/` | Unity capture scripts used to produce the reference renders. |
| `filter_documentation.html` | Plain-language documentation of every filter and parameter (static export of the tuner's `/docs` page). |

## 🚀 Quick start

### Install

Requires Python 3.12 or newer.

```bash
pip install vipsim                  # CPU reference (numpy, pillow)
pip install "vipsim[teichopsia]"    # + scikit-image, needed by apply_teichopsia
pip install "vipsim[gpu]"           # + torch, for vipsim.gpu / VipSimGPU
pip install "vipsim[gym]"           # + torch and gymnasium, for VisualImpairmentWrapper
pip install "vipsim[all]"           # everything
```

From a checkout of this repository: `pip install -e ".[all]"`.

### Command line

```bash
# Single filter (`python -m vipsim ...` and `python vipsim_filters.py ...` work too)
vipsim input.png output.png --filter blur --severity 0.5
vipsim input.png output.png --filter cvd --type deuteranomaly --severity 0.8
vipsim input.png output.png --filter cataracts --severity 0.5

# Brightness / contrast / gamma
vipsim input.png output.png --filter bcg --brightness 0.8 --contrast 0.7 --gamma 1.5

# Named preset (a chain of filters)
vipsim input.png output.png --preset moderate_low_vision

# See everything available
vipsim --list-presets
```

### Python API

```python
from PIL import Image
from vipsim import apply_cvd, apply_preset

img = Image.open("scene.png").convert("RGB")

# Apply one filter
out = apply_cvd(img, cvd_type="deuteranomaly", severity=0.6)

# Apply a clinical preset (chained filters)
out = apply_preset(img, "macular_degeneration")

out.save("scene_impaired.png")
```

Or as a module: `import vipsim as vip`, then `vip.apply_blur(img, 0.5)`, `vip.PRESETS`, `vip.__version__`. Every filter is exported as `apply_<name>`.

### GPU / batched

The PyTorch port works on a whole batch of frames at once and keeps everything on-device. Every filter is a `VipSimGPU` method with the CPU name and keyword arguments (`sim.cvd(x, cvd_type=..., severity=...)` is `apply_cvd(img, cvd_type=..., severity=...)` on a batch). Image-independent data (warp fields, index maps, masks, lookup tables) is computed once per parameter set and frame size by the CPU's own NumPy code and kept in a bounded on-device cache.

Needs `pip install "vipsim[gpu]"`. `VipSimGPU` and `VisualImpairmentWrapper` can also be imported straight from `vipsim`; torch is only loaded when you first touch them, so a plain `import vipsim` stays fast and works without torch.

```python
import json
import torch
from vipsim import VipSimGPU
from vipsim.gpu import frames_to_gpu, gpu_to_frames

sim = VipSimGPU(device="cuda")
x = frames_to_gpu(frames, sim.device)           # frames: [N, H, W, 3] uint8 -> [N, 3, H, W] in [0,1]

x = sim.cvd(x, cvd_type="deuteranomaly", severity=0.8)
x = sim.blur(x, severity=0.4)
x = sim.apply_preset(x, "macular_degeneration")
x = sim.apply_profile(x, json.load(open("vipsim_profiles_v2/p3.json")))
out = gpu_to_frames(x)                          # back to uint8, same pixels as the CPU chain
```

### Inside an RL / MyoSuite environment

```python
from vipsim import VisualImpairmentWrapper    # needs vipsim[gym]

env = VisualImpairmentWrapper(
    env,
    chain=[("cvd", {"severity": 0.8}), ("blur", {"severity": 0.4})],
    device="cuda",
    image_key=None,        # set this if the observation is a dict
)
```

`chain` takes any filter with its CPU keyword arguments; `vipsim.PRESETS[name]["filters"]` is a ready chain, and `vipsim.gpu.profile_chain(profile)` turns a `vipsim_profiles_v2` profile into one.

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

Run `vipsim --list-presets` for the full set of 24 with descriptions.

## 🧑‍🤝‍🧑 Participant profiles (`vipsim_profiles_v2/`)

The VIP-Sim paper validated its shader stacks with 7 low-vision participants
(Table 4) and shows the same street scene through each participant's eyes
(Figure 6). This repo carries recreations of those 7 profiles as filter stacks:
one filter per validated shader, calibrated against clean extractions of the
Figure-6 panels. Each `pN.json` documents its Table-4 mapping and carries the
stack itself; `renders/` holds paper-vs-ours comparisons. Similarity vs the
paper panels (mean of SSIM, pixel, and CIEDE2000 color): 81 to 98 percent.

Tune or inspect them interactively:

```bash
pip install scipy scikit-image      # tuner extras
python validation_scripts/profile_tuner.py    # then open http://localhost:8765
```

The tuner shows the reference next to a live render, scores the similarity as
you drag sliders, and can auto-tune all numeric values (coordinate descent).
Every filter and parameter is explained in plain language at `/docs` (also
exported as `filter_documentation.html`). Steps may carry `"_disabled": true`
to switch them off without deleting them; all scripts in this repo honor that.

## 🔬 Under the hood (GPU port)

All 21 filters of `vipsim.filters` (the catalogue above plus `vignette`) are ported. The CPU filters are the reference: each GPU twin reproduces today's CPU output, quirks included (nearest-index truncation, 8-bit intermediates, Pillow's integer arithmetic), and has been **bit-exact** with it in every comparison so far, on CUDA and on CPU torch, with Pillow 12.2 and 13. The CPU only ever sees a uint8 image, so each twin reads its input as 8-bit levels and returns `level / 255`: a chain of GPU filters sees exactly the CPU chain's images. One exception concerns batches: a stochastic filter seeded with a shared `numpy.random.Generator` (or `None`) draws one field per frame, in batch order, so a batched chain of two such filters consumes the draws filter by filter (it equals running each CPU filter over the whole batch in turn, not the CPU chain frame by frame). Fixed integer seeds, one Generator per filter, or one frame per call (as in the wrapper) avoid this.

There are no custom kernels. The backend is built from a few primitives:

1. **Per-level tables and channel mixes.** Every step that maps one channel value to another (sRGB ↔ linear, `pow`, brightness/contrast/gamma, posterize, the floaters' blend) is the CPU's own NumPy code, evaluated once on the 256 levels and looked up on the device; torch's `pow` is about 1 ulp off NumPy's, which is enough to flip 8-bit levels. Three-channel mixes (the `cvd` matrices, luminance) add their products in the order NumPy's BLAS does.
2. **Pillow-exact blur and resize.** `pil_gaussian_blur` replays `ImageFilter.GaussianBlur` (three extended-box passes per axis in 24-bit fixed point, each rounded to 8 bits) with cumulative sums in integers, so its cost does not grow with the radius. `VipSimGPU.pil_resize` replays Pillow's BILINEAR and NEAREST resamplers, including the installed Pillow's pass order (12.2 and 13 differ). Neither uses `conv2d`, which runs in TF32 by default on Ampere and later GPUs. They serve `blur`, `bloom`, the `field_loss` pyramid and `pixelation`.
3. **Pixel gathers.** `gather_nearest` and `gather_bilinear` replace NumPy fancy indexing for `distortion`, `wiggle`, `vortex`, the cataract frost and the `led` cells; slices and rolls cover `double_vision`, `nystagmus` and `glitch`.

Anything that doesn't depend on the image content (index maps, masks, noise fields, tables) is computed **once** per parameter set and frame size by the CPU's NumPy code, seeded RNG included, and kept in a bounded LRU cache on the device, so per-frame work is a few batched tensor operations. Cache keys are exact and typed: under NumPy 2, `np.float64(0.5)` and `0.5` can give different CPU results, so they never share an entry.

**What is checked.**

- `tests/test_gpu_*.py` compare every GPU filter with its CPU filter pixel for pixel. They use random and natural frames at several sizes (odd, portrait, 1 pixel wide and very tall), severities near 0 and 1, every keyword argument (NumPy scalar types included), batches of different frames, other input dtypes, cache reuse, and the CPU's exceptions for invalid input.
- `tests/test_gpu_primitives.py` checks the blur and resize against Pillow itself and the gathers against NumPy indexing.
- `tests/test_gpu_profiles.py` renders all 7 profiles (as `render_profiles_v2.py` does) on both backends step by step and end to end, and all 24 presets end to end.
- `validation_scripts/validate_gpu_vs_numpy.py` runs every ported filter on the 2048×1024 scene and exits with an error if one exceeds its documented tolerance (`vipsim.gpu.TWIN_TOLERANCE`; results in `abs_diff_out/gpu_vs_numpy_summary.*`) or is skipped for a missing optional dependency (`--allow-skip` accepts that). The tolerance is 0 levels for every filter except `cvd`, `bcg` and `foveal_darkness`, which may differ by 1 level on another BLAS build, because it can sum their channel mix in a different order. The tolerance is per filter, so `foveal_darkness` gets it in every mode, although only `mode="desat"` mixes channels.

Because the GPU port equals the NumPy port, it inherits the NumPy port's validation against Unity (below).

**Speed.** In the port's verification runs on an RTX 4050 Laptop GPU, one 2048×1024 frame took 3–30 ms per filter (0.1–0.2 s for `field_loss`), against 0.01–2.5 s for the NumPy filter. A batch of 64 frames of 84×84 cost about as much as one frame for most filters; a single small frame is launch-bound and no faster than NumPy. `pixelation` is the exception: like the CPU, it resamples to a grid of cells that does not shrink with the frame (about 510 cells across at the default severity, 800 at severity 0.2), so a batch of 64 frames of 84×84 takes about 0.1–0.2 s and up to 2 GB of GPU memory. The first call with new parameters or a new frame size runs the CPU code to build the cached fields: up to about 1.5 s at 2048×1024 for `vortex`. The cache is bounded by entries, not bytes, so a sweep over many parameter sets at full resolution can hold several GB of fields.

## ✅ Validation

The Python port isn't just visually plausible — it's checked against the Unity original. Two complementary methods are used, documented in full in [`vipsim_equivalence_report.md`](https://github.com/Max-Raed/VIP-Sim_Port/blob/main/vipsim_equivalence_report.md):

- **Method A — algorithmic proof.** For pure deterministic arithmetic filters (`bcg`, `cvd`), the HLSL fragment math and the NumPy code are lined up and shown to compute the same function. Fed Unity's own unfiltered capture, both match the Unity render to within 1 level (MAE 0.14 for bcg, 0.07 for cvd).
- **Method B — measured fidelity.** For everything else (noise, kernels, halo scatter, UV warps, gaze-contingent and time-varying effects), matched 2048×1024 captures are compared through a stack of published metrics: **MAE, SSIM, SSIM_blur, ΔE2000, Wasserstein-1, LPIPS**.

A filter passes when its best sweep point reaches **SSIM_blur ≥ 0.90** against the matched Unity reference. SSIM_blur is SSIM after a σ = 4 px Gaussian pre-blur of both images (`validation_scripts/metric_constants.py`), which discounts detail finer than about 10 px.

For the three inherently time-varying filters (`flickering_stars`, `wiggle`, `nystagmus`) a **burst-frame protocol** captures 30 matched frames over a fixed window and aggregates the metric across them.

**Status: 17 of 19 tracked filters pass.** `glitch` sits just below threshold (SSIM_blur 0.890, non-deterministic) and `led` is marginal pending cell-phase tuning.

**Caveats (audit, 2026-10-29).** A pass is weaker evidence than it looks:

- The unfiltered image already scores SSIM_blur ≥ 0.90 against 12 of the 17 passing Unity references, so for those filters the criterion does not separate the port from doing nothing. Of the passing filters only `bcg`, `cataracts` and `double_vision` are clearly separated; `cvd` rests on its Method A proof instead.
- The best sweep point is chosen on the same reference it is scored against; nothing is held out.
- The Unity references were rendered from a block-compressed (DXT/BC1) copy of the input, so Unity and Python start from images that differ by MAE ≈ 2.6. That offset, not fp16 rounding, is the ~3 MAE "floor" in the results table.
- The `flickering_stars` and `nystagmus` burst validators test their own time-aware models, not `apply_flickering_stars` / `apply_nystagmus`.
- `field_loss` has no source-level proof and is now listed under Method B.

### Reproduce the sweep

This needs a clone of the repository (the Unity captures and scripts are not in the PyPI package). `./validate.sh` does all of it, including the virtualenv; by hand:

```bash
source .venv/bin/activate
python validation_scripts/validate_all_filters.py --no-strips

# burst-frame validators for the time-varying filters
python validation_scripts/validate_flickering_stars_burst.py
python validation_scripts/validate_wiggle_burst.py
python validation_scripts/validate_nystagmus_burst.py
```

### Tests

```bash
python -m unittest discover -s tests -v
python validation_scripts/validate_gpu_vs_numpy.py   # GPU vs NumPy on the 2048x1024 scene
```

The GPU tests run on CUDA when it is available, else on CPU torch, and are skipped without torch.

### Releasing to PyPI

The distribution name is `vipsim`; the version lives in `vipsim/__init__.py` (`__version__`), which the build reads.

Releases are published by [`.github/workflows/publish.yml`](https://github.com/Max-Raed/VIP-Sim_Port/blob/main/.github/workflows/publish.yml) through PyPI Trusted Publishing, so no API token is involved:

1. Bump `__version__` in `vipsim/__init__.py` and commit.
2. Create a GitHub release with a matching tag (e.g. `v0.1.0`). Publishing the release builds the sdist and wheel, checks them with `twine check`, and uploads them to PyPI.

To check a build locally before releasing:

```bash
python -m pip install build twine
python -m build                      # writes dist/vipsim-<version>.tar.gz and .whl
python -m twine check dist/*
```

## 📚 References

- Wang, Bovik, Sheikh & Simoncelli (2004). *Image quality assessment: from error visibility to structural similarity.* IEEE TIP 13(4).
- Sharma, Wu & Dalal (2005). *The CIEDE2000 color-difference formula.* Color Research & Application 30(1).
- Zhang, Isola, Efros, Shechtman & Wang (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR.
- Kantorovich (1942) / Villani (2008) — optimal transport (Wasserstein distance).

## 📄 License & attribution

VipSim is released under the **MIT License** (see [`LICENSE`](https://github.com/Max-Raed/VIP-Sim_Port/blob/main/LICENSE)).

It reimplements shader code from **[VIP-Sim](https://github.com/Max-Raed/VIP-Sim)** by Max Raed, which is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); please keep that attribution when you use or redistribute VipSim.

---

<div align="center">
<sub>Built for accessibility research and embodied-AI evaluation.</sub>
</div>