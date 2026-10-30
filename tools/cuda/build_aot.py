#!/usr/bin/env python3
"""Build one AOT artifact set: cubins plus a manifest. Never executes device code.

The artifact set is a build product. `make build-kernels-aot ARCH=sm_89 CUDA_HOME=...` writes it
under `target/kernels/aot/`; CI packages it as a sidecar. The manifest records the toolchain, the
contract and every cubin digest so the loader can reject a mismatched or corrupted file.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# One entry per kernel the AOT profile can ship today. `source` is relative to the repository.
KERNELS = [
    {
        "name": "projection_dense_f32",
        "contract": "canonical-f32",
        "source": "crates/backend/cuda/kernels/aot/projection.cu",
        "cubin": "projection.cubin",
        "flags": ["--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"],
    },
    {
        "name": "attention_fp8_256",
        "contract": "fp8-kv-f32acc",
        "source": "crates/backend/cuda/kernels/shared-attention/attention-sm120.cu",
        "cubin": "attention.cubin",
        "flags": ["--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"],
    },
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toolchain(nvcc):
    output = subprocess.check_output([str(nvcc), "--version"], text=True)
    match = re.search(r"release [\d.]+, (V[\d.]+)", output)
    if not match:
        raise SystemExit("cannot parse nvcc version")
    return f"nvcc {match.group(1).lstrip('V')}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="sm_89", help="target architecture, e.g. sm_89")
    parser.add_argument("--out", type=Path, default=Path("target/kernels/aot"), help="output root")
    parser.add_argument("--cuda-home", type=Path, help="CUDA toolkit root (default: CUDA_HOME)")
    parser.add_argument("--plan", action="store_true", help="print the plan and exit")
    arguments = parser.parse_args()
    cuda_home = arguments.cuda_home or Path(os.environ.get("CUDA_HOME", "/opt/cuda"))
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise SystemExit(f"nvcc not found under {cuda_home}")
    plan = {
        "profile": "aot",
        "arch": arguments.arch,
        "toolchain": toolchain(nvcc),
        "out": str(arguments.out),
        "kernels": [k["name"] for k in KERNELS],
    }
    if arguments.plan:
        print(json.dumps(plan, indent=2))
        return
    target = arguments.out / arguments.arch
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing artifact set: {target}")
    target.mkdir(parents=True)
    entries = []
    for kernel in KERNELS:
        source = REPO / kernel["source"]
        cubin = target / kernel["cubin"]
        command = [
            str(nvcc),
            "--cubin",
            f"-arch={arguments.arch}",
            "-O3",
            *kernel["flags"],
            str(source),
            "-o",
            str(cubin),
        ]
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
        entries.append(
            {
                "name": kernel["name"],
                "contract": kernel["contract"],
                "cubin": kernel["cubin"],
                "sha256": sha256(cubin),
            }
        )
    manifest = {
        "schema": 1,
        "profile": "aot",
        "arch": arguments.arch,
        "toolchain": plan["toolchain"],
        "entries": entries,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
