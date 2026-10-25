"""Every workspace boundary states why it exists.

The build plan keeps two kinds of boundary - the main Rust workspace and the reference
workspaces that must not reach the production lock or resolution graph - and says a new
independent workspace has to justify its dependency or toolchain isolation rather than appear
for a few helper lines, one test or a copied lint table.

This checks that each independent workspace is registered with that reason, that it really is
independent (its own lockfile, not published, not a member of the root workspace), and that
the reason list only shrinks.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTERED = {
    "tools/bench/containers/Cargo.toml": (
        "candidate containers: bbtree/Verus and hashbrown comparisons stay outside the "
        "production lock and dependency graph; benchmarks use actual production wrappers"
    ),
    "crates/backend/cuda/kernels/oxide-attention/Cargo.toml": (
        "device-only Rust SIMT artifact: pins nightly-2026-10-11 and the NVIDIA cuda-oxide "
        "Git SDK; host builds remain stable and consume the verified embedded PTX"
    ),
    "crates/backend/cuda/kernels/oxide-gdn/Cargo.toml": (
        "device-only Rust SIMT artifact: pins nightly-2026-10-11 and the NVIDIA cuda-oxide "
        "Git SDK; host builds remain stable and consume the verified embedded PTX"
    ),
    "tools/bench/attention/Cargo.toml": (
        "heavy reference framework: the Candle attention comparison pins its own CUDA-enabled "
        "candle crates and must stay out of the production lock and resolution graph"
    ),
}
SKIP = {"target", ".git", "artifacts"}


def manifests(root=ROOT):
    for path in sorted(Path(root).rglob("Cargo.toml")):
        if any(part in SKIP for part in path.relative_to(root).parts):
            continue
        yield path


def independent(root=ROOT):
    """Manifests declaring their own workspace, excluding the root's."""
    for path in manifests(root):
        if path.parent == Path(root):
            continue
        document = tomllib.loads(path.read_text())
        if "workspace" in document:
            yield path


def check(root=ROOT, registered=None):
    registered = dict(REGISTERED if registered is None else registered)
    root_manifest = tomllib.loads((Path(root) / "Cargo.toml").read_text())
    members = set(root_manifest["workspace"]["members"])
    seen = set()
    problems = []
    for path in independent(root):
        name = str(path.relative_to(root))
        reason = registered.get(name)
        if reason is None:
            problems.append(
                f"{name}: an independent workspace with no recorded reason; state the dependency "
                "or toolchain isolation it exists for, or fold it into the main workspace"
            )
            continue
        seen.add(name)
        if len(reason) < 20:
            problems.append(f"{name}: the reason has to say something")
        if not (path.parent / "Cargo.lock").exists():
            problems.append(f"{name}: declares its own workspace without a lockfile")
        if tomllib.loads(path.read_text()).get("package", {}).get("publish") is not False:
            problems.append(f"{name}: a reference workspace is not published")
        if any(str(path.parent.relative_to(root)) == member for member in members):
            problems.append(f"{name}: is also a member of the root workspace")
    stale = set(registered) - seen
    if stale:
        problems.append(f"drop the entries that are no longer independent: {sorted(stale)}")
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"Workspace boundaries: {len(seen)} independent workspace(s), each with a reason")


if __name__ == "__main__":
    check()
