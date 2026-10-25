"""Cargo/Apple-linker and cargo-zigbuild archive driver; target policy belongs to build.rs."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def output(command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def rust_host():
    return output(["rustc", "-vV"]).split("host: ", 1)[1].splitlines()[0]


def planner():
    # Same build.rs is both Cargo's build script and the packaging contract.
    source = ROOT / "build.rs"
    key = hashlib.sha256(
        source.read_bytes()
        + (ROOT / "tools/build/backend_cfg.rs").read_bytes()
        + output(["rustc", "-vV"]).encode()
    ).hexdigest()[:16]
    binary = ROOT / "target/package" / f"build-contract-{key}"
    if not binary.exists():
        binary.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=binary.parent) as temporary:
            staged = Path(temporary) / "contract"
            run(["rustc", "--edition", "2024", str(source), "-o", str(staged)])
            staged.replace(binary)
    return binary


def resolve(name="auto", target=None):
    return json.loads(output([str(planner()), "--plan", target or rust_host(), name]))


def zigbuild_version():
    binary = shutil.which("cargo-zigbuild") or str(
        Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")) / "bin/cargo-zigbuild"
    )
    return output([binary, "--version"])


def preflight(plan):
    run([str(planner()), "--preflight", plan["zig_target"], rust_host()])
    zigbuild_version()
    output([os.environ.get("CARGO_ZIGBUILD_ZIG_PATH", "zig"), "version"])


def build_env(plan):
    return {**os.environ, "INFER_PACKAGE_BUILD": "1", "INFER_PACKAGE_TARGET": plan["zig_target"]}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cargo(action, plan):
    # Apple clang supports both macOS architectures. Use its native linker for cdylib
    # dependencies: Zig loses rustc's exported-symbols-list argument on this path.
    if (
        action == "zigbuild"
        and plan["target"].endswith("-apple-darwin")
        and platform.system() == "Darwin"
    ):
        action = "build"
    command = [
        "cargo",
        action,
        "--locked",
        "--target",
        plan["zig_target"] if action == "zigbuild" else plan["target"],
        "-p",
        "infer-cli",
        "--no-default-features",
    ]
    if plan["features"]:
        command.extend(["--features", ",".join(plan["features"])])
    return command


def target_validation(plan):
    """The validation a package run performs for its own target.

    The host suites (layout, policy, formatting, clippy and the workspace tests) run in the Rust
    and tools jobs of the same pipeline, so repeating them here would multiply the most expensive
    job by the number of package targets. A cross target still compiles its tests, because that is
    the only place it is checked; a host target is covered by the binary build and smoke below.
    """
    if plan["target"] == rust_host():
        return "host-target-build-and-smoke"
    run([*cargo("zigbuild", plan), "--tests", "--release"], env=build_env(plan))
    return "cross-cli-tests-compiled-not-executed"


def native_cargo(plan):
    """Cargo for a native build: the plan's features and package, without a cross target."""
    command = ["cargo", "build", "--locked", "-p", "infer-cli", "--no-default-features"]
    if plan["features"]:
        command.extend(["--features", ",".join(plan["features"])])
    return command


def native_env(plan):
    """The environment a native build needs, discovered in the same place as the preflight.

    `BINDGEN_EXTRA_CLANG_ARGS` is the discovery the Makefile used to hand-roll per platform:
    bindgen cannot find the host C standard headers on its own, and the path must not be copied
    from another machine. Reporting what was found keeps the failure legible when it is absent.
    """
    env = {**os.environ, "INFER_PACKAGE_BUILD": "1"}
    if plan["backend"] != "cuda":
        return env
    include = output(["gcc", "-print-file-name=include"]).strip()
    if include and (Path(include) / "stddef.h").is_file():
        existing = env.get("BINDGEN_EXTRA_CLANG_ARGS", "")
        if include not in existing:
            env["BINDGEN_EXTRA_CLANG_ARGS"] = f"{existing} -isystem {include}".strip()
    return env


def native(plan):
    """`make local-build`: the host release CLI, from the same plan and preflight as packaging."""
    preflight(plan)
    run([*native_cargo(plan), "--release"], env=native_env(plan))
    return f"native-{plan['backend']}-release-cli"


def test(plan):
    """`make test`: run the host checks and suites, plus a target test compile for cross targets."""
    run(["python3", "tools/check/layout.py"])
    run(["python3", "tools/check/policy.py"])
    run(["python3", "-m", "unittest", "discover", "-s", "tools/package", "-p", "test_*.py"])
    run(["cargo", "fmt", "--all", "--check"])
    if plan["target"] == rust_host():
        run([*cargo("clippy", plan), "--all-targets", "--", "-D", "warnings"], env=build_env(plan))
        run(cargo("test", plan), env=build_env(plan))
        return "native-cli-tests"
    run([*cargo("zigbuild", plan), "--tests", "--release"], env=build_env(plan))
    return "cross-cli-tests-compiled-not-executed"


def build(plan):
    result = run(
        [*cargo("zigbuild", plan), "--release", "--bin", "infer", "--message-format=json"],
        stdout=subprocess.PIPE,
        text=True,
        env=build_env(plan),
    )
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    binaries = [
        row["executable"]
        for row in rows
        if row.get("reason") == "compiler-artifact"
        and row.get("target", {}).get("name") == "infer"
        and row.get("executable")
    ]
    metadata = [
        Path(row["out_dir"]) / "infer-build.json"
        for row in rows
        if row.get("reason") == "build-script-executed"
        and (Path(row["out_dir"]) / "infer-build.json").is_file()
    ]
    if len(binaries) != 1 or len(metadata) != 1:
        raise ValueError("Cargo did not report exactly one binary and build.rs manifest")
    if json.loads(metadata[0].read_text()) != plan:
        raise ValueError("compiled build.rs contract differs from requested release")
    binary = Path(binaries[0])
    check_binary(binary, plan)
    return binary


def check_binary(binary, plan):
    with binary.open("rb") as stream:
        header = stream.read(32)
    if plan["backend"] == "cuda":
        if header[:6] != b"\x7fELF\x02\x01" or len(header) < 20:
            raise ValueError("expected a 64-bit little-endian ELF executable")
        machine = int.from_bytes(header[18:20], "little")
        expected = {"x86_64": 62, "aarch64": 183}[plan["arch"]]
    else:
        if header[:4] != b"\xcf\xfa\xed\xfe" or len(header) < 8:
            raise ValueError("expected a 64-bit little-endian Mach-O executable")
        machine = int.from_bytes(header[4:8], "little")
        expected = {"x86_64": 0x1000007, "aarch64": 0x100000C}[plan["arch"]]
    if machine != expected:
        raise ValueError("binary CPU architecture differs from release target")


def inspect_archive(archive, destination):
    """Reject traversal, links, duplicate files and checksum mismatches before smoke testing."""
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(
            not m.isfile() or Path(m.name).is_absolute() or ".." in Path(m.name).parts
            for m in members
        ):
            raise ValueError("archive must contain unique, relative regular files")
        roots = {Path(name).parts[0] for name in names}
        if len(roots) != 1:
            raise ValueError("archive must have exactly one package root")
        bundle.extractall(destination, filter="data")
    root = destination / roots.pop()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != 2:
        raise ValueError("unsupported package manifest schema")
    expected = manifest["files"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if set(expected) | {"manifest.json"} != actual:
        raise ValueError("archive file inventory differs from manifest")
    for name, checksum in expected.items():
        if digest(root / name) != checksum:
            raise ValueError(f"checksum mismatch: {name}")
    plan = resolve(manifest["platform"], manifest["zig_target"])
    if any(manifest[key] != value for key, value in plan.items()):
        raise ValueError("archive build contract mismatch")
    if "bin/infer" not in expected or not os.access(root / "bin/infer", os.X_OK):
        raise ValueError("missing executable bin/infer")
    check_binary(root / "bin/infer", plan)
    return root, manifest


def smoke(root):
    # Run outside the checkout: the installed CLI cannot rely on repository-relative assets.
    binary = root / "bin/infer"
    for argument in ("--version", "--help"):
        subprocess.run([binary, argument], cwd=root, check=True, stdout=subprocess.PIPE)


def verify(archive, model=None, golden=None):
    if not archive or not Path(archive).is_file():
        raise ValueError("PACKAGE_FILE must name an existing .tar.gz archive")
    with tempfile.TemporaryDirectory(prefix="infer-package-check-") as temporary:
        root, manifest = inspect_archive(Path(archive), Path(temporary))
        native = manifest["target"] == rust_host()
        if native:
            smoke(root)
        else:
            print("Cross-target archive: integrity/architecture verified; execution not attempted.")
        if model is not None:
            if not native:
                raise ValueError("GPU acceptance must run on the package target host")
            if not model or not golden or not Path(model).is_dir() or not Path(golden).is_file():
                raise ValueError("GPU acceptance requires MODEL directory and GOLDEN file")
            command = [
                str(root / "bin/infer"),
                "--backend",
                manifest["backend"],
                "verify",
                "--package",
                str(Path(model).resolve()),
                "--golden",
                str(Path(golden).resolve()),
            ]
            if manifest["platform"] == "linux-cuda":
                command = ["bash", "tools/bench/safe-run.sh", *command]
            run(command)
    print(f"Verified: {archive}", flush=True)


def package(plan, out):
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    label = f"infer-{version}-{plan['zig_target']}-{plan['backend']}"
    final = out / label
    if final.exists():
        raise ValueError(f"refusing to overwrite existing release: {final}")
    validation = target_validation(plan)
    binary = build(plan)
    with tempfile.TemporaryDirectory(prefix=".package-", dir=out) as temporary:
        stage = Path(temporary) / label
        (stage / "bin").mkdir(parents=True)
        shutil.copy2(binary, stage / "bin/infer")
        for source in ("LICENSE", "Cargo.lock"):
            shutil.copy2(ROOT / source, stage / Path(source).name)
        backend = plan["backend"]
        requirements = (
            "Linux with compatible libc, NVIDIA driver and CUDA/cuTile JIT toolchain."
            if backend == "cuda"
            else "Compatible macOS with a Metal-capable GPU."
        )
        (stage / "README.md").write_text(
            f"# infer {version} ({plan['zig_target']}, {backend})\n\n"
            f"Requirements: {requirements}\n"
            "Models, drivers and toolchains are not bundled.\n\n"
            "```sh\n./bin/infer --help\n"
            f"./bin/infer --backend {backend} serve --package /path/to/model "
            "--listen 127.0.0.1:8080\n```\n\n"
            "See manifest.json for build provenance and file checksums. "
            "Package smoke tests do not certify GPU inference or performance.\n"
        )
        manifest = {
            **plan,
            "version": version,
            "rustc": output(["rustc", "-vV"]),
            "build_system": platform.platform(),
            "host_libc": platform.libc_ver(),
            "host": rust_host(),
            "cargo_zigbuild": zigbuild_version(),
            "zig": output([os.environ.get("CARGO_ZIGBUILD_ZIG_PATH", "zig"), "version"]),
            "commit": output(["git", "rev-parse", "HEAD"]),
            "dirty": bool(output(["git", "status", "--porcelain"])),
            "validation": [validation, "binary-architecture", "archive-integrity"],
            # The source checks this package relies on instead of repeating them per target: the
            # `commit` and `dirty` above are what tie the archive to their results.
            "source_checks": ["make check-rust", "make check-tools"],
            "smoke_executed": plan["target"] == rust_host(),
            "gpu_inference_accepted": False,
            "files": {
                str(p.relative_to(stage)): digest(p)
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            },
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        publication = Path(temporary) / "publication"
        publication.mkdir()
        archive = publication / f"{label}.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    bundle.add(path, arcname=str(path.relative_to(stage.parent)), recursive=False)
        verify(archive)
        (publication / "SHA256SUMS").write_text(f"{digest(archive)}  {archive.name}\n")
        shutil.copy2(stage / "manifest.json", publication / "manifest.json")
        publication.rename(final)
    print(f"Package: {final / archive.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["build", "test", "package", "verify", "accept", "native"]
    )
    parser.add_argument("--platform", default="auto")
    parser.add_argument("--target", default="")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/packages")
    parser.add_argument("--archive")
    parser.add_argument("--model")
    parser.add_argument("--golden")
    args = parser.parse_args()
    if args.action in ("verify", "accept"):
        if args.action == "accept" and (not args.model or not args.golden):
            raise ValueError("GPU acceptance requires --model and --golden")
        verify(args.archive, args.model if args.action == "accept" else None, args.golden)
    else:
        plan = resolve(args.platform, args.target)
        preflight(plan)
        if args.action == "package":
            package(plan, args.out)
        elif args.action == "build":
            print(build(plan))
        elif args.action == "native":
            print(native(plan))
        else:
            test(plan)


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        tarfile.TarError,
    ) as error:
        raise SystemExit(str(error)) from error
