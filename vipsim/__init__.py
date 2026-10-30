"""VipSim: visual-impairment simulation filters.

A NumPy/PIL port of the VIP-Sim Unity shaders (https://github.com/Max-Raed/VIP-Sim).
The PyTorch backend lives in `vipsim.gpu` and is not imported here, so
`import vipsim` does not require torch.
"""

__version__ = "0.1.0"

from .filters import (  # noqa: F401
    CVD_MATRICES,
    PRESETS,
    apply_bcg,
    apply_bloom,
    apply_blur,
    apply_cataracts,
    apply_cvd,
    apply_detail_loss,
    apply_distortion,
    apply_double_vision,
    apply_field_loss,
    apply_flickering_stars,
    apply_floaters,
    apply_foveal_darkness,
    apply_glitch,
    apply_led,
    apply_noise,
    apply_nystagmus,
    apply_pixelation,
    apply_preset,
    apply_teichopsia,
    apply_vignette,
    apply_vortex,
    apply_wiggle,
)

__all__ = [
    "CVD_MATRICES",
    "PRESETS",
    "apply_bcg",
    "apply_bloom",
    "apply_blur",
    "apply_cataracts",
    "apply_cvd",
    "apply_detail_loss",
    "apply_distortion",
    "apply_double_vision",
    "apply_field_loss",
    "apply_flickering_stars",
    "apply_floaters",
    "apply_foveal_darkness",
    "apply_glitch",
    "apply_led",
    "apply_noise",
    "apply_nystagmus",
    "apply_pixelation",
    "apply_preset",
    "apply_teichopsia",
    "apply_vignette",
    "apply_vortex",
    "apply_wiggle",
]
