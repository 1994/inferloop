#!/usr/bin/env python3
"""Build one AOT artifact set: cubins plus a manifest. Never executes device code.

The artifact set is a build product. `make build-kernels-aot ARCH=sm_89 CUDA_HOME=...` writes it
under `target/kernels/aot/`. The `sidecar` section of `kernels.json` lists the entries and pins one
toolchain per architecture; a different nvcc is refused, because a cubin from a newer toolchain is
not evidence for the driver that architecture ships with. The set appears only once every cubin has
built.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).resolve().with_name("kernels.json")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nvcc_version(nvcc):
    output = subprocess.check_output([str(nvcc), "--version"], text=True)
    match = re.search(r"release [\d.]+, V([\d.]+)", output)
    if not match:
        raise SystemExit("cannot parse nvcc version")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="sm_89", help="target architecture, e.g. sm_89")
    parser.add_argument("--out", type=Path, default=Path("target/kernels/aot"), help="output root")
    parser.add_argument("--cuda-home", type=Path, help="CUDA toolkit root (default: CUDA_HOME)")
    arguments = parser.parse_args()
    document = json.loads(REGISTRY.read_text())
    registry = document["sidecar"]
    pinned = registry["toolchains"].get(arguments.arch)
    if pinned is None:
        raise SystemExit(f"no toolchain pinned for {arguments.arch} in {REGISTRY.name}")
    cuda_home = arguments.cuda_home or Path(os.environ.get("CUDA_HOME", "/opt/cuda"))
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise SystemExit(f"nvcc not found under {cuda_home}")
    version = nvcc_version(nvcc)
    if version != pinned and not version.startswith(f"{pinned}."):
        raise SystemExit(f"{arguments.arch} requires nvcc {pinned}, found {version} in {cuda_home}")
    target = arguments.out / arguments.arch
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing artifact set: {target}")
    staging = arguments.out / f".{arguments.arch}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    entries = []
    for kernel in registry["kernels"]:
        source = REPO / kernel["source"]
        cubin = staging / kernel["cubin"]
        command = [
            str(nvcc),
            "--cubin",
            f"-arch={arguments.arch}",
            "-O3",
            *registry["flags"],
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
                "source": kernel["source"],
                "source_sha256": sha256(source),
                "flags": registry["flags"],
                "cubin": kernel["cubin"],
                "sha256": sha256(cubin),
            }
        )
    manifest = {
        "schema": document["schema"],
        "profile": "aot",
        "arch": arguments.arch,
        "toolchain": f"nvcc {version}",
        "entries": entries,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(target)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
