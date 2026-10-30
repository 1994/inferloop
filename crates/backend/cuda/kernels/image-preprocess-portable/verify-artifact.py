#!/usr/bin/env python3
"""Compatibility entry point; the kernel catalog owns artifact verification."""

from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    path = Path(__file__).resolve()
    sys.argv = [str(path), "verify", "--kernel", path.parent.name, *sys.argv[1:]]
    sys.path.insert(0, str(path.parents[5] / "tools/cuda"))
    runpy.run_path(str(path.parents[5] / "tools/cuda/kernels.py"), run_name="__main__")
