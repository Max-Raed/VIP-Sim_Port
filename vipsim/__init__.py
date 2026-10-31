"""VipSim: visual-impairment simulation filters.

A NumPy/PIL port of the VIP-Sim Unity shaders (https://github.com/Max-Raed/VIP-Sim).
The PyTorch backend lives in `vipsim.gpu`. `VipSimGPU` and
`VisualImpairmentWrapper` are also reachable from here, but only load torch
on first access, so `import vipsim` neither requires torch nor pays for
importing it.
"""

__version__ = "0.1.1"

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

# Re-exported from vipsim.gpu on first access (PEP 562): torch is optional and
# slow to import. Not in __all__, so `from vipsim import *` never needs torch.
_GPU_EXPORTS = ("VipSimGPU", "VisualImpairmentWrapper")


def __getattr__(name):
    if name in _GPU_EXPORTS:
        try:
            from . import gpu
        except ImportError as e:
            raise ImportError(
                f"vipsim.{name} needs PyTorch: pip install 'vipsim[gpu]' ({e})"
            ) from e
        return getattr(gpu, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_GPU_EXPORTS))
