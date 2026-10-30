"""Compatibility alias for in-repo scripts: the code lives in `vipsim/filters.py`.

`import vipsim_filters` returns the `vipsim.filters` module itself (private
names included), and `python vipsim_filters.py ...` runs its CLI. Not shipped
in the PyPI package; installed users import `vipsim`.
"""
import sys

from vipsim import filters as _filters

if __name__ == "__main__":
    _filters.main()
else:
    sys.modules[__name__] = _filters
