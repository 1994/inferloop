#!/usr/bin/env python3
"""One offline entry point for embedded CUDA artifacts; never executes device code."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from fa4 import rebuild_f16

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "crates/backend/cuda/kernels"
CATALOG = Path(__file__).with_name("kernels.json")
SHARED = re.compile(r"__shared_mem_([a-f0-9]{16})_([0-9]+)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog():
    data = json.loads(CATALOG.read_text())
    require(data["schema"] == 1, "Unsupported kernel catalog schema")
    entries = data["kernels"]
    names = [e["name"] for e in entries]
    require(len(names) == len(set(names)), "Duplicate kernel registration")
    discovered = {p.parent.name for p in ROOT.glob("*/verify-artifact.py")}
    require(set(names) == discovered, "Kernel catalog and artifact directories differ")
    registered = set()
    for entry in entries:
        require(entry["builder"] in {"oxide", "nvcc", "fa4", "fa4-f16"}, "Unknown kernel builder")
        directory = ROOT / entry["name"]
        registered.add(directory / entry["ptx"])
        # FA4 contains both a bridge and architecture-specific generated PTX.
        manifest = directory / "manifest.json"
        if manifest.exists():
            document = json.loads(manifest.read_text())
            for variant in document["entries"]:
                registered.update(
                    directory / variant["name"] / f for f in variant["files"] if f.endswith(".ptx")
                )
    require(set(ROOT.rglob("*.ptx")) <= registered, "Unregistered embedded PTX")
    for entry in data["sidecar"]["kernels"]:
        require((REPO / entry["source"]).is_file(), f"Missing sidecar source: {entry['name']}")
    return data


def normalize(text, entry):
    matches = SHARED.findall(text)
    require(len({namespace for namespace, _ in matches}) == 1, "Expected one shared namespace")
    declared = re.findall(
        r"^\.visible \.shared \.align 4 \.b8 __shared_mem_[a-f0-9]{16}_([0-9]+)\[([0-9]+)\];$",
        text,
        re.M,
    )
    expected = [(str(i), str(size)) for i, size in enumerate(entry["shared_bytes"])]
    require(sorted(declared) == expected, "Shared allocation layout differs")
    require({i for _, i in matches} == {i for i, _ in expected}, "Unexpected shared reference")
    return SHARED.sub(lambda match: "__shared_mem_CANONICAL_" + match.group(2), text)


def check_hash(path, expected):
    require(sha(path) == expected, f"Identity mismatch: {path}")


def verify(entry, generated=None):
    directory = ROOT / entry["name"]
    provenance = (
        json.loads((directory / "provenance.json").read_text())
        if (directory / "provenance.json").exists()
        else {}
    )
    ptx = directory / entry["ptx"]
    source = directory / entry["source"]
    verified_files = set()
    if "source_sha256" in provenance:
        check_hash(source, provenance["source_sha256"])
        check_hash(ptx, provenance["ptx_sha256"])
        verified_files.update((source, ptx))
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for file, digest in manifest.get("common_files", {}).items():
            check_hash(directory / file, digest)
            verified_files.add(directory / file)
        for variant in manifest["entries"]:
            base = directory if entry["name"].endswith("f16") else directory / variant["name"]
            for file, digest in variant["files"].items():
                check_hash(base / file, digest)
                verified_files.add(base / file)
    require(
        {source, ptx} <= verified_files, "Missing source/PTX identity in provenance or manifest"
    )
    text = ptx.read_text()
    require(
        re.search(r"\.target\s+" + re.escape(entry["target"]) + r"\b", text),
        "Unexpected PTX target",
    )
    symbols = re.findall(r"\.visible\s+\.entry\s+(\w+)\(", text)
    require(sorted(symbols) == sorted(entry["symbols"]), "Unexpected PTX entry points")
    if entry["builder"] == "oxide":
        lock = tomllib.loads((directory / "Cargo.lock").read_text())
        toolchain = tomllib.loads((directory / "rust-toolchain.toml").read_text())
        dependencies = tomllib.loads((directory / "Cargo.toml").read_text())["dependencies"]
        require(
            toolchain["toolchain"]["channel"] == provenance["toolchain"], "Unpinned Rust toolchain"
        )
        for name in ("cuda-core", "cuda-device", "cuda-host"):
            require(
                dependencies[name].get("rev") == provenance["cuda_rust_commit"]
                and dependencies[name].get("git") == "https://github.com/NVIDIA/cuda-rust",
                f"Unpinned SDK manifest: {name}",
            )
            packages = [p for p in lock["package"] if p["name"] == name]
            require(
                len(packages) == 1
                and packages[0].get("source", "").endswith("#" + provenance["cuda_rust_commit"]),
                f"Unpinned SDK: {name}",
            )
        normalize(text, entry)
    if entry["builder"] != "oxide":
        for symbol in entry["symbols"]:
            require(source.read_text().count(symbol + "(") == 1, "Unexpected device source entry")
    if "arguments" in provenance:
        require(provenance["arguments"] == entry["arguments"], "Compile contract differs")
    if "canonical_attention_ptx_sha256" in provenance:
        base = ROOT / (
            "vector-attention" if entry["name"].endswith("combined") else "oxide-attention"
        )
        check_hash(base / "attention-sm120.ptx", provenance["canonical_attention_ptx_sha256"])
        if "base_source_sha256" in provenance:
            check_hash(base / "attention-sm120.cu", provenance["base_source_sha256"])
            require(
                provenance["base_ptx_sha256"] == provenance["canonical_attention_ptx_sha256"],
                "Base PTX identities diverged",
            )
    # Frozen FA4 verifiers additionally inspect ABI, numerical profile and ELF dependencies.
    # Preserve these receipts; they are themselves part of the AOT manifest identity.
    abi = "not-applicable"
    if entry["name"].startswith("fa4-"):
        if entry["name"].endswith("f32") or sys.platform.startswith("linux"):
            env = {**os.environ, "PYTHONOPTIMIZE": "0"}
            subprocess.run(
                [sys.executable, str(directory / "verify-artifact.py")],
                check=True,
                env=env,
                stdout=subprocess.DEVNULL,
            )
            abi = "verified"
        else:
            abi = "requires-linux-readelf"
    if generated is not None:
        actual = generated.read_text()
        if entry["builder"] == "oxide":
            require(
                normalize(text, entry) == normalize(actual, entry),
                "Regenerated PTX differs beyond shared symbol renaming",
            )
        else:
            require(
                ptx.read_bytes() == generated.read_bytes(), "Regenerated PTX is not byte-identical"
            )
    return {
        "kernel": entry["name"],
        "ptx_sha256": sha(ptx),
        "regenerated": generated is not None,
        "abi": abi,
    }


def run(command, cwd, log, env=None):
    command = list(map(str, command))
    environment = {
        **(os.environ if env is None else env),
        "PYTHONOPTIMIZE": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MAX_JOBS": "1",
        "NVCC_THREADS": "1",
    }
    receipt = {"argv": command, "cwd": str(cwd)}
    log.with_suffix(".command.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with log.open("w") as output:
        result = subprocess.run(
            command, cwd=cwd, env=environment, stdout=output, stderr=subprocess.STDOUT
        )
    receipt["returncode"] = result.returncode
    log.with_suffix(".command.json").write_text(json.dumps(receipt, indent=2) + "\n")
    require(result.returncode == 0, f"Command failed ({result.returncode}); see {log}")


def rebuild(entry, args, version):
    verify(entry)
    directory = ROOT / entry["name"]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = {
        "inputs": {
            str(p.relative_to(REPO)): sha(p)
            for p in [
                CATALOG,
                Path(__file__),
                Path(__file__).with_name("fa4.py"),
                Path(__file__).with_name("compile_f16.py"),
                directory / entry["source"],
                *[
                    p
                    for p in (directory / "provenance.json", directory / "manifest.json")
                    if p.exists()
                ],
            ]
        },
        "kernel": entry["name"],
        "scope": entry["rebuild_scope"],
        "completed": False,
        "GPU_execution": False,
    }
    try:
        if entry["builder"] == "oxide":
            require(
                args.oxide_backend is not None,
                "--oxide-backend must identify the pinned compiler library",
            )
            provenance = json.loads((directory / "provenance.json").read_text())
            check_hash(args.oxide_backend, provenance["compiler_backend_sha256"])
            build = out / "source"
            shutil.copytree(
                directory, build, ignore=shutil.ignore_patterns("target", "__pycache__")
            )
            env = {
                **os.environ,
                "CUDA_OXIDE_BACKEND": str(args.oxide_backend.resolve()),
                "CARGO_TARGET_DIR": str(out / "target"),
            }
            command = [
                args.cargo,
                "oxide",
                "build",
                "--arch",
                "sm_120",
                "--no-fmad",
            ]
            run(command, build, out / "build.log", env)
            check_hash(build / "Cargo.lock", sha(directory / "Cargo.lock"))
            package = tomllib.loads((build / "Cargo.toml").read_text())["package"]["name"]
            generated = build / (package + ".ptx")
        else:
            nvcc = args.cuda_root.resolve() / "bin/nvcc"
            compiler = subprocess.check_output([str(nvcc), "--version"], text=True)
            require(
                re.search(r"V" + re.escape(version) + r"(?:\s|$)", compiler),
                f"Expected nvcc {version}",
            )
            record["compiler"] = compiler
            if entry["builder"] == "fa4-f16":
                generated = rebuild_f16(entry, args, directory, out, run, check_hash, require)
            elif entry["builder"] == "fa4":
                require(
                    args.upstream is not None and args.sdk_site is not None,
                    "FA4 requires --upstream and --sdk-site",
                )
                command = [
                    args.python,
                    directory / "regenerate.py",
                    "--upstream",
                    args.upstream.resolve(),
                    "--sdk-site",
                    args.sdk_site.resolve(),
                    "--cuda-root",
                    args.cuda_root.resolve(),
                    "--cxx",
                    args.cxx,
                    "--output",
                    out / "aot",
                ]
                env = dict(os.environ)
                extra = [str(p.resolve()) for p in (args.quack, args.tvm_ffi) if p is not None]
                env["PYTHONPATH"] = os.pathsep.join([*extra, env.get("PYTHONPATH", "")])
                run(command, directory, out / "build.log", env)
                # The existing generator verifies its new bundle. Compare reviewed outputs too.
                manifest = json.loads((directory / "manifest.json").read_text())
                for variant in manifest["entries"]:
                    for file, digest in variant["files"].items():
                        check_hash(out / "aot" / variant["name"] / file, digest)
                generated = out / "aot/bridge.ptx"
            else:
                generated = out / entry["ptx"]
                command = [
                    nvcc,
                    *entry["arguments"],
                    source_path := directory / entry["source"],
                    "-o",
                    generated,
                ]
                record["source_sha256"] = sha(source_path)
                run(command, directory, out / "build.log")
        record["verification"] = verify(entry, generated)
        record["completed"] = True
    except Exception as error:
        record["error"] = str(error)
        raise
    finally:
        record["commands"] = [json.loads(p.read_text()) for p in sorted(out.glob("*.command.json"))]
        (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "verify", "rebuild"])
    parser.add_argument(
        "--kernel", action="append", help="Registered name; repeat to select several"
    )
    parser.add_argument("--generated", type=Path, help="Compare one freshly generated PTX")
    parser.add_argument(
        "--output", type=Path, help="New rebuild directory; never overwrites a receipt"
    )
    parser.add_argument(
        "--cuda-root", type=Path, default=Path(os.environ.get("CUDA_TOOLKIT_PATH", "/opt/cuda"))
    )
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--quack", type=Path)
    parser.add_argument("--tvm-ffi", type=Path)
    parser.add_argument("--oxide-backend", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--sdk-site", type=Path)
    args = parser.parse_args()
    data = catalog()
    selected = data["kernels"]
    if args.kernel:
        require(set(args.kernel) <= {e["name"] for e in selected}, "Unknown kernel")
        selected = [e for e in selected if e["name"] in args.kernel]
    if args.generated:
        require(
            args.action == "verify" and len(selected) == 1,
            "--generated needs verify and one kernel",
        )
    if args.action == "rebuild":
        require(
            len(selected) == 1 and args.output is not None,
            "Rebuild needs one --kernel and --output",
        )
        version = selected[0].get("nvcc_version", data["nvcc_version"])
        print(json.dumps(rebuild(selected[0], args, version)))
    elif args.action == "verify":
        results = [verify(e, args.generated) for e in selected]
        print(json.dumps({"verified": len(results), "results": results}, indent=2))
    else:
        print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
