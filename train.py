"""Thin wrapper around the canonical 3DPS training entry point.

This delegates to ``mm3DPS.train_frame_nvs.main`` so callers can run::

    python train.py --scene seq_0_frame_135 --target_n 20000 --iters 500

instead of::

    python -m mm3DPS.train_frame_nvs --scene seq_0_frame_135 ...

Both invocations are equivalent.
"""

from __future__ import annotations

import os
import runpy
import sys


def _ensure_package_on_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


def main() -> None:
    _ensure_package_on_path()
    runpy.run_module("mm3DPS.train_frame_nvs", run_name="__main__")


if __name__ == "__main__":
    main()
