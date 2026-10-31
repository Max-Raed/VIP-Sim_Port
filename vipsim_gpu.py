"""Compatibility alias for in-repo scripts: the code lives in `vipsim/gpu.py`.

`import vipsim_gpu` returns the `vipsim.gpu` module itself. Not shipped in
the PyPI package; installed users import `vipsim.gpu`.
"""
import sys

from vipsim import gpu as _gpu

if __name__ == "__main__":
    import runpy
    runpy.run_module("vipsim.gpu", run_name="__main__")
else:
    sys.modules[__name__] = _gpu
