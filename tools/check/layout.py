"""Keep crate ownership, API facades, dependency direction, and local links coherent."""

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {"foundation", "backend", "model", "engine", "diagnostics", "service", "testing"}
# Development tools live outside the responsibility groups: they are members so they share the
# workspace's lock, versions and lints, but they are not part of the product crate layout.
TOOL_MEMBERS = {"tools/bench/cpu"}
# Device-only artifact builders pin a separate nightly CUDA backend and Git SDK.
# They are reviewed by workspaces.py and never enter the stable host resolution graph.
DEVICE_BUILD_MEMBERS = {
    "crates/backend/cuda/kernels/oxide-attention",
    "crates/backend/cuda/kernels/oxide-gdn",
}
SERVICE = {"infer-frontdoor", "infer-agent", "infer-cli"}
BACKENDS = {
    "infer-backend-metal",
    "infer-backend-cuda",
}
# These lexical checks protect the device boundary; they are not a complete Rust parser or a
# substitute for reviewing new hardware APIs. Artifact eligibility may consume reported facts.
CUDA_HARDWARE_PROBES = re.compile(
    r"\b(?:cuDeviceGet\w*|cuDeviceComputeCapability|cuDeviceTotalMem\w*|"
    r"cuDriverGetVersion|cudaDeviceGet\w*|cudaGetDevice\w*|"
    r"cuda(?:Driver|Runtime)GetVersion|nvmlDeviceGet\w*|"
    r"nvjpegGetHardware\w*|nvjpegDecoderJpegSupported|nvjpegDecodeBatchedSupported\w*)\b"
    r"|\btarget_(?:os|arch|env|family|vendor|feature|pointer_width)\b"
    r"|\bcfg(?:_attr)?!?\s*\([\s\w(),]*\b(?:unix|windows)\b"
    r"|\b(?:std::)?env::consts::(?:OS|ARCH)\b"
    r"|/proc/driver/nvidia/version"
    r"|(?i:\b(?:lib(?:nvjpeg\w*|cublas(?:lt)?|cudart|cuda)|"
    r"(?:nvjpeg|cublas(?:lt)?|cudart|cuda)\w*)\.(?:so(?:\.\d+)*|dylib|dll)\b)"
)


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def check_cuda_hardware_boundary(root=ROOT):
    """Keep CUDA hardware/platform probes and library names in the device boundary."""
    source = root / "crates/backend/cuda/src"
    device = source / "device"
    require(source.is_dir() and device.is_dir(), "CUDA device boundary is missing")
    for path in source.rglob("*.rs"):
        if path == source / "device.rs" or path.is_relative_to(device):
            continue
        match = CUDA_HARDWARE_PROBES.search(path.read_text())
        if match:
            raise SystemExit(
                f"{path.relative_to(root)}: move CUDA hardware/platform detection "
                f"or library name {match.group()!r} into the device boundary"
            )


def dependency_tables(manifest):
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        yield manifest.get(section, {})
    for target in manifest.get("target", {}).values():
        yield from dependency_tables(target)


def check_ownership(member, dependencies):
    """Production dependencies must preserve model, compiler and scheduler ownership."""
    internal = {name for name in dependencies if name.startswith("infer-")}
    foundation = {"infer-core", "infer-ir", "infer-spi"}
    contracts = {"infer-gpu-api", "infer-kernel-api"}
    if member.startswith("crates/foundation/") or member in {
        "crates/model/recipes",
        "crates/engine/state",
        "crates/engine/scheduler",
        "crates/backend/api",
        "crates/backend/kernel-api",
    }:
        require(
            not internal - foundation,
            f"{member}: keep contracts and state independent of implementations",
        )
    if member == "crates/model/compiler":
        require(
            not internal - foundation - contracts,
            f"{member}: compile supplied IR; do not construct model recipes",
        )
    if member == "crates/model/package":
        require(
            not internal - foundation - {"infer-model-recipes"},
            f"{member}: model providers must not depend on compilation or execution",
        )
    if member == "crates/engine/workloads":
        # I1's boundary: the algorithm providers describe plans, candidates, sampling and state
        # needs over the shared IR/SPI and the model description. A device backend, its kernel API
        # or the GPU API would make this layer the place per-algorithm branches accumulate, which is
        # what moving the policies out of CUDA is meant to end.
        require(
            not internal - foundation - {"infer-models", "infer-model-recipes"},
            f"{member}: algorithms plan over the shared contracts, not over a device backend",
        )
    if member == "crates/engine/runtime":
        require(
            not internal & {"infer-models", "infer-model-recipes"},
            f"{member}: orchestrate the bound graph; do not import model recipes",
        )
        # I1: the runtime coordinates lifecycles, budgets and fairness over the contracts; a device
        # executor in its production dependencies is what makes one algorithm's execution the
        # runtime's concern. Test executors are dev-dependencies and are not considered here.
        require(
            not internal & BACKENDS,
            f"{member}: orchestrate over the contracts; do not depend on a device executor",
        )


def production_dependencies(manifest):
    dependencies = set(manifest.get("dependencies", {})) | set(
        manifest.get("build-dependencies", {})
    )
    for target in manifest.get("target", {}).values():
        dependencies.update(production_dependencies(target))
    return dependencies


def check_crates():
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]
    members = set(workspace["members"])
    require(members >= TOOL_MEMBERS, f"Unregistered tool members: {sorted(TOOL_MEMBERS - members)}")
    found = {str(path.parent.relative_to(ROOT)) for path in (ROOT / "crates").rglob("Cargo.toml")}
    product = members - TOOL_MEMBERS
    expected = product | DEVICE_BUILD_MEMBERS
    require(found == expected, f"Unregistered or missing crates: {sorted(found ^ expected)}")
    for member in sorted(product):
        directory = ROOT / member
        parts = Path(member).parts
        require(len(parts) >= 3 and parts[1] in GROUPS, f"{member}: choose a responsibility group")
        require(
            len(parts) == (4 if parts[1] == "testing" else 3),
            f"{member}: use group/crate, or testing/cpu/crate",
        )
        if parts[1] == "testing":
            require(parts[2] == "cpu", f"{member}: CPU reference backends belong in testing/cpu")
        manifest = tomllib.loads((directory / "Cargo.toml").read_text())
        check_ownership(member, production_dependencies(manifest))
        dependencies = set()
        for table in dependency_tables(manifest):
            dependencies.update(table)
            for name, value in table.items():
                if isinstance(value, dict) and "path" in value:
                    target = directory / value["path"] / "Cargo.toml"
                    require(target.is_file(), f"{member}: broken path dependency {name}")
        if parts[1] != "service":
            require(
                not dependencies & SERVICE, f"{member}: implementation must not depend on services"
            )
        if parts[1] == "engine":
            require(
                not set(manifest.get("dependencies", {})) & BACKENDS,
                f"{member}: depend on backend contracts, not concrete executors",
            )
        facade = directory / "src/lib.rs"
        if facade.is_file():
            source = facade.read_text()
            require(len(source.splitlines()) <= 100, f"{member}: keep lib.rs a small API facade")
            require(
                not re.search(
                    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s|^impl\b", source, re.M
                ),
                f"{member}: place implementation in responsibility modules",
            )
    for name, value in workspace["dependencies"].items():
        if isinstance(value, dict) and "path" in value:
            require(
                (ROOT / value["path"] / "Cargo.toml").is_file(), f"Broken workspace path: {name}"
            )
    print(f"Layout: {len(members)} grouped crates, thin facades and dependency boundaries passed")


def check_links():
    paths = [
        ROOT / "README.md",
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "crates").rglob("*.md"),
        *(ROOT / "tools").rglob("*.md"),
        *(ROOT / "examples").rglob("*.md"),
    ]
    for path in paths:
        # Generated validation artifacts may be absent in a fresh checkout.
        for match in re.finditer(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            link = unquote(match[1].split("#", 1)[0].strip("<>"))
            if not link or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", link):
                continue
            target = (path.parent / link).resolve()
            if target.is_relative_to(ROOT / "artifacts"):
                continue
            require(target.exists(), f"{path.relative_to(ROOT)}: broken local link {link}")
    print("Documentation: local navigation links passed")


def check_sources():
    for path in (ROOT / "crates").rglob("*.rs"):
        for source in re.findall(r'file:\s*"(crates/[^"\n]+)"', path.read_text()):
            require(
                (ROOT / source).is_file(), f"{path.relative_to(ROOT)}: stale source map {source}"
            )
        for fixture in re.findall(
            r'env!\("CARGO_MANIFEST_DIR"\)[^;]{0,80}?\.join\("([^"\n]+)"\)',
            path.read_text(),
        ):
            directory = next(parent for parent in path.parents if (parent / "Cargo.toml").is_file())
            require(
                (directory / fixture).exists(),
                f"{path.relative_to(ROOT)}: broken manifest-relative fixture {fixture}",
            )
    print("Observability and fixtures: source paths passed")


if __name__ == "__main__":
    check_crates()
    check_cuda_hardware_boundary()
    check_links()
    check_sources()
