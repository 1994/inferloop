"""Check lint inheritance, exceptions, deployment feature isolation, and magic numbers."""

import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TILE_KERNELS = ROOT / "crates/backend/cuda-kernels/src/tile"
API_ROOT = ROOT / "crates/service/frontdoor/src/http"
# Wire parsing and opaque protocol payloads are the only places a dynamically shaped JSON value
# is unavoidable. Every response object and SSE event must use a typed struct instead.
API_VALUE_ALLOWED = {
    "openai/request.rs": "OpenAI wire request DTOs",
    "openai/responses/request.rs": "Responses wire request DTOs and opaque tool schemas",
    "openai/decisions/request.rs": "Decisions wire request DTOs",
    "observability.rs": "observation payload owned by infer-observe",
}
WORKSPACE = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def dependency_tables(manifest):
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        yield manifest.get(section, {})
    for target in manifest.get("target", {}).values():
        yield from dependency_tables(target)


def check_manifests():
    config = tomllib.loads((ROOT / "clippy.toml").read_text())
    require(
        config.get("too-many-lines-threshold", 100) <= 100,
        "Function length limit must stay at 100 or below",
    )
    require(
        config.get("cognitive-complexity-threshold", 25) <= 25,
        "Cognitive complexity limit must stay at 25 or below",
    )
    # Reference candidates keep independent resolution graphs and the same strict lints.
    for harness in ("attention", "containers"):
        benchmark = tomllib.loads((ROOT / f"tools/bench/{harness}/Cargo.toml").read_text())
        require(
            benchmark["lints"] == WORKSPACE["lints"],
            f"{harness} harness must keep all strict lint gates",
        )
    inherited = set()
    for member in WORKSPACE["members"]:
        manifest = tomllib.loads((ROOT / member / "Cargo.toml").read_text())
        require(manifest.get("lints") == {"workspace": True}, f"{member}: inherit workspace lints")
        require(
            "test-backends" not in manifest.get("features", {}).get("default", []),
            f"{member}: CPU test features must be opt-in",
        )
        for table in dependency_tables(manifest):
            inherited.update(
                name
                for name, value in table.items()
                if isinstance(value, dict) and value.get("workspace")
            )
    unused = set(WORKSPACE["dependencies"]) - inherited
    require(not unused, f"Unused workspace dependencies: {sorted(unused)}")
    # The other direction: a member declares category, target, optional and features, and takes the
    # version from the root table. Without this a member can quietly reintroduce a second version
    # source that no check notices.
    for member in WORKSPACE["members"]:
        manifest = tomllib.loads((ROOT / member / "Cargo.toml").read_text())
        for table in dependency_tables(manifest):
            for name, value in table.items():
                require(
                    name in WORKSPACE["dependencies"],
                    f"{member}: {name} is not declared in workspace.dependencies",
                )
                require(
                    not (isinstance(value, dict) and "version" in value),
                    f"{member}: {name} declares its own version; inherit the workspace entry",
                )
    for group in ("all", "pedantic", "nursery"):
        require(WORKSPACE["lints"]["clippy"][group]["level"] == "deny", f"Keep {group} denied")
    for lint in (
        "unwrap_used",
        "expect_used",
        "panic",
        "todo",
        "unimplemented",
        "dbg_macro",
        "undocumented_unsafe_blocks",
        "allow_attributes_without_reason",
        "mem_forget",
        "multiple_unsafe_ops_per_block",
        "unused_result_ok",
        "get_unwrap",
        "rc_mutex",
        "too_many_lines",
        "cognitive_complexity",
    ):
        require(WORKSPACE["lints"]["clippy"][lint] == "deny", f"Keep {lint} denied")
    require(WORKSPACE["lints"]["rust"]["warnings"] == "deny", "Keep Rust warnings denied")
    require(WORKSPACE["lints"]["rust"]["unsafe_code"] == "deny", "Keep unsafe code denied")
    require(
        WORKSPACE["lints"]["rust"]["unfulfilled_lint_expectations"] == "deny",
        "Expired lint expectations must fail the build",
    )


def check_exceptions():
    pattern = re.compile(r"#!?\[(.*?)\]", re.DOTALL)
    forbidden = {
        "warnings",
        "clippy::all",
        "clippy::pedantic",
        "clippy::nursery",
        "clippy::too_many_lines",
        "clippy::cognitive_complexity",
    }
    paths = [
        *(ROOT / "crates").rglob("*.rs"),
        *(ROOT / "tools/bench/cpu/src").rglob("*.rs"),
        *(ROOT / "tools/bench/attention/src").rglob("*.rs"),
        *(ROOT / "tools/bench/containers/src").rglob("*.rs"),
    ]
    for path in paths:
        for attribute in pattern.finditer(path.read_text()):
            text = attribute[1]
            match = re.search(r"\b(?:allow|expect)\s*\((.*?)\)", text, re.DOTALL)
            if match is None:
                continue
            reason = re.search(r'reason\s*=\s*"([^"]+)"', text)
            require(reason is not None and len(reason[1]) >= 20, f"{path}: explain lint exceptions")
            lints = {part.strip() for part in match[1].split(",") if "reason" not in part}
            require(not lints & forbidden, f"{path}: strict lint checks cannot be waived")
            if "unsafe_code" in lints:
                require(
                    path
                    in (
                        ROOT / "crates/model/package/src/storage/safetensors.rs",
                        ROOT / "crates/backend/metal/src/device.rs",
                        ROOT / "crates/backend/cuda/src/device.rs",
                        ROOT / "tools/bench/attention/src/device.rs",
                        ROOT / "crates/backend/cuda/src/device/readback.rs",
                        # Gather boundary validates every tap, shape and table address before
                        # recording and retains all device owners until completion.
                        TILE_KERNELS / "vision/vision_position_kernels.rs",
                        ROOT / "crates/backend/cuda/src/device/vision_position.rs",
                        # RGB8 PTX boundary checks all tensor extents and retains the source,
                        # coefficient tables, intermediate, output and modules until the fence.
                        ROOT / "crates/backend/cuda/src/device/image_preprocess/launch.rs",
                        # The no-resize PTX boundary retains its input and output until the fence.
                        ROOT / "crates/backend/cuda/src/device/image_preprocess/direct.rs",
                        # nvJPEG's optional FFI retains decoder state and stream inputs to the
                        # completion fence and binds every handle to its owning device context.
                        ROOT / "crates/backend/cuda/src/device/image_preprocess/nvjpeg.rs",
                        ROOT / "crates/backend/cuda/src/device/cublaslt.rs",
                        ROOT / "crates/backend/cuda/src/device/vendor_fp8.rs",
                        # FP4 extension of the Lt ABI: checked complete extents, exact output
                        # proofs, zero workspace, one owning stream and retained handles/weights.
                        ROOT / "crates/backend/cuda/src/device/vendor_fp8/nvfp4.rs",
                        # PDL launch boundaries: checked extents, owning stream and retained
                        # consumers until the completion fence.
                        TILE_KERNELS / "mlp/pdl_kernels.rs",
                        TILE_KERNELS / "mlp/pdl_consumers_kernels.rs",
                        ROOT / "crates/backend/cuda/src/mlp/pdl.rs",
                        ROOT / "crates/backend/cuda/src/mlp/pdl_consumers.rs",
                        # Borrowed tensor boundary: host-owned pointers are wrapped for exactly
                        # one synchronized launch and dropped before returning.
                        TILE_KERNELS / "projection/contract.rs",
                        # AOT driver boundary: every FFI call is checked, handles are owned and
                        # buffers are freed exactly once.
                        ROOT / "crates/backend/cuda-kernels/src/aot/driver.rs",
                        ROOT / "crates/backend/cuda/src/resident/pool_gemv.rs",
                        # Raw-address launch contract: obligations are stated on the trait and
                        # discharged by the owning caller until the host adopts contract buffers.
                        ROOT / "crates/backend/cuda-kernels/src/contract/kernels.rs",
                        ROOT / "crates/backend/cuda-kernels/src/aot/projection.rs",
                        ROOT / "crates/backend/cuda-kernels/examples/aot-pool-gemv.rs",
                        ROOT / "crates/backend/cuda/src/resident/arena.rs",
                        ROOT / "crates/backend/cuda/src/resident/prefill_burst.rs",
                        # Narrow callers of lifetime-bound FP4 arena aliases: sole-consumer
                        # product, exclusive arena lease, completion and same-stream replay.
                        ROOT
                        / "crates/backend/cuda/src/resident/nvfp4_gemm/workspace/vendor_fp4.rs",
                        ROOT / "crates/backend/cuda/src/resident/profile.rs",
                        # Explicit PTX launch boundaries: checked extents, owning device stream,
                        # Function/Module retention and input/output resource leases.
                        ROOT / "crates/backend/cuda/src/resident/attention_oxide.rs",
                        # AOT boundary: embedded hash-verified ABI, checked tensor extents,
                        # exact owning stream and retained libraries/modules/scratch.
                        ROOT / "crates/backend/cuda/src/resident/attention_mma/host.rs",
                        ROOT / "crates/backend/cuda/src/resident/attention_mma/mod.rs",
                        ROOT / "crates/backend/cuda/src/resident/attention_mma/recorded.rs",
                        ROOT / "crates/backend/cuda/src/resident/recurrent_prefill/oxide.rs",
                        # Conversion measurement: fixed buffers and access leases;
                        # synchronize on the owning stream before releasing storage.
                        ROOT / "crates/backend/cuda/tests/unit/tuning_attention_fp8_convert.rs",
                        ROOT / "crates/backend/cuda/src/resident/batch.rs",
                        # Slot state aliases: source pools outlive programs and captured
                        # graphs; only the current lease holder replays on the device stream.
                        ROOT / "crates/backend/cuda/src/resident/slot_batch.rs",
                        ROOT / "crates/backend/cuda/src/loading/mod.rs",
                        # Homes drop before their source pools and clear before pool eviction.
                        ROOT / "crates/backend/cuda/src/executor/execution.rs",
                        # Disjoint verification output rows and staged commit inputs retain
                        # their owning arena for the lifetime of every captured graph.
                        ROOT / "crates/backend/cuda/src/resident/slot_recurrent.rs",
                        # Draft graph replays: engine-owned buffers and rings on the device
                        # stream, fenced by the token readback or drained before teardown.
                        ROOT / "crates/backend/cuda/src/resident/dflash2/mod.rs",
                        ROOT / "crates/backend/cuda/src/resident/slot_verify.rs",
                        ROOT / "crates/foundation/core/src/placement/mod.rs",
                        ROOT / "tools/bench/cpu/src/main.rs",
                        ROOT / "tools/bench/containers/src/main.rs",
                    ),
                    f"{path}: unsafe code belongs only at audited device, "
                    "file mapping or measurement boundaries",
                )
    deny = tomllib.loads((ROOT / "deny.toml").read_text())
    for exception in deny["advisories"].get("ignore", []) + deny["bans"].get("skip", []):
        require(
            isinstance(exception, dict) and exception.get("reason"), "Explain dependency exceptions"
        )


MAGIC_NUMBER = re.compile(
    r"(?<![\w\[])(?<![\w\)\]]\.)(?:0x[0-9a-fA-F_]+|0b[01_]+|0o[0-7_]+|\d[\d_]*(?:\.\d[\d_]*)?"
    r"(?:[eE][+-]?\d+)?)(?:f32|f64|i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize)?"
)
MAGIC_ARRAY_HEAD = re.compile(
    r"\[\s*(0x[0-9a-fA-F_]+|0b[01_]+|0o[0-7_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?)"
    r"(?:f32|f64|i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize)?\b"
)
MAGIC_WHITELIST = {"0", "1", "2", "0.0", "1.0", "2.0"}
MAGIC_LITERAL_OPENERS = set("([{=,:;!&|+-*/<>")
MAGIC_SUFFIX = re.compile(r"(?:f32|f64|i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize)$")
MAGIC_CONST_ITEM = re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+(?!fn\b)[A-Za-z_]")
MAGIC_STRING = re.compile(r'"(?:\\.|[^"\\])*"')
MAGIC_CHAR = re.compile(r"b?'(?:\\.|[^'\\])'")
MAGIC_TEST_PARTS = {"tests", "examples", "benches"}
STRATEGY_SOURCE = ROOT / "crates/backend/cuda/src/strategy.rs"
HARDWARE_BRANCH = re.compile(r"NVIDIA|GeForce|RTX|\bsm_\d|(?:bf16|fp8-channel|nvfp4):\d+x\d+")


def magic_number_base(token):
    return MAGIC_SUFFIX.sub("", token).replace("_", "")


def magic_masked_lines(lines):
    """Blank out test/DSL blocks and block comments before scanning a file."""
    masked = list(lines)
    index = 0
    while index < len(lines):
        if "#[cfg(test)]" in lines[index] or "#[cutile::module]" in lines[index]:
            depth = 0
            started = False
            end = index
            while end < len(lines):
                for char in lines[end]:
                    if char == "{":
                        depth += 1
                        started = True
                    elif char == "}":
                        depth -= 1
                masked[end] = ""
                if started and depth == 0:
                    break
                if not started and "{" not in lines[end] and ";" in lines[end]:
                    break
                end += 1
            index = end + 1
        else:
            index += 1
    in_block_comment = False
    for lineno, line in enumerate(masked):
        if in_block_comment:
            end = line.find("*/")
            if end < 0:
                masked[lineno] = ""
                continue
            line = masked[lineno] = line[end + 2 :]
            in_block_comment = False
        start = line.find("/*")
        if start >= 0 and "*/" not in line[start + 2 :]:
            in_block_comment = True
            masked[lineno] = line[:start]
    return masked


def magic_number_lines(path):
    """Yield production-code lines with tests, DSL modules, and const items blanked out."""
    lines = magic_masked_lines(path.read_text().splitlines())
    in_const = False
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        if MAGIC_CONST_ITEM.match(stripped):
            in_const = True
        if in_const:
            if stripped.endswith(";"):
                in_const = False
            continue
        code = line.split("//", 1)[0]
        yield lineno, MAGIC_CHAR.sub("''", MAGIC_STRING.sub('""', code)), stripped


def magic_number_files():
    """Production Rust sources under every workspace crate's `src` directory."""
    for path in sorted((ROOT / "crates").rglob("*.rs")):
        relative = path.relative_to(ROOT)
        if "src" not in relative.parts:
            continue
        name = path.name
        if name == "tests.rs" or name.endswith(("_tests.rs", "_check.rs")):
            continue
        if MAGIC_TEST_PARTS.intersection(relative.parts):
            continue
        yield path


def magic_number_violations(relative, lineno, code, stripped):
    """Yield the magic-number violations found on one production source line."""
    for match in MAGIC_NUMBER.finditer(code):
        if magic_number_base(match[0]) not in MAGIC_WHITELIST:
            yield f"{relative}:{lineno}: {match[0]} in `{stripped[:80]}`"
    for match in MAGIC_ARRAY_HEAD.finditer(code):
        if magic_number_base(match[1]) in MAGIC_WHITELIST:
            continue
        before = code[: match.start()].rstrip()
        if before and before[-1] not in MAGIC_LITERAL_OPENERS:
            continue
        yield f"{relative}:{lineno}: {match[1]} in `{stripped[:80]}`"


def check_magic_numbers():
    """Production Rust code must name its numeric literals."""
    violations = []
    for path in magic_number_files():
        relative = path.relative_to(ROOT)
        for lineno, code, stripped in magic_number_lines(path):
            violations.extend(magic_number_violations(relative, lineno, code, stripped))
    require(
        not violations,
        "Name numeric literals instead of writing magic numbers:\n" + "\n".join(violations),
    )


def check_strategy_neutrality():
    """Tile choice is a measurement, so no policy may branch on a device name or model shape."""
    lines = magic_masked_lines(STRATEGY_SOURCE.read_text().splitlines())
    for lineno, line in enumerate(lines, 1):
        code = line.split("//", 1)[0]
        require(
            HARDWARE_BRANCH.search(code) is None,
            f"{STRATEGY_SOURCE.relative_to(ROOT)}:{lineno}: record this tile in the tuning table "
            "or measure it with autotune instead of branching on hardware or a model shape: "
            f"`{line.strip()[:80]}`",
        )
    print("Strategy: tile selection stays independent of device names and model shapes")


ALGORITHM_NAMES = re.compile(r"\b(mtp|dflash2?|replayssm)[A-Za-z0-9_]*", re.IGNORECASE)
COMMON_LAYERS = (
    "crates/foundation/ir/src",
    "crates/foundation/spi/src",
    "crates/engine/workloads/src",
)


# I1's removal list: the common layers still name MTP in these symbols. The rule is a ratchet, so
# the list may only shrink - the migration that removes a symbol also removes its entry, and a stale
# entry fails rather than lingering.
ALGORITHM_NAME_BASELINE = {}


def check_algorithm_neutrality(root=ROOT, baseline=None):
    """The common layers describe drafts in general, not one algorithm.

    I1's split puts the algorithm in its provider; the reason to hold the line is that a name in a
    common layer is how an executor grows a branch per algorithm. Comments and test modules may name
    them - the rule is about code that branches - and the symbols I1 has not removed yet are listed
    above so the coupling can only shrink.
    """
    baseline = dict(ALGORITHM_NAME_BASELINE if baseline is None else baseline)
    for relative in COMMON_LAYERS:
        for path in sorted((root / relative).rglob("*.rs")):
            location = str(path.relative_to(root))
            lines = magic_masked_lines(path.read_text().splitlines())
            for lineno, line in enumerate(lines, 1):
                code = line.split("//", 1)[0]
                match = ALGORITHM_NAMES.search(code)
                if match is None:
                    continue
                symbol = match.group(0)
                if symbol in baseline.get(location, set()):
                    continue
                snippet = line.strip()[:80]
                raise SystemExit(
                    f"{location}:{lineno}: {relative} must not name a speculation algorithm "
                    f"(`{symbol}`); keep it in the provider that implements it: `{snippet}`"
                )
    stale = {
        location: symbols - _symbols_in(root / location, symbols)
        for location, symbols in baseline.items()
    }
    stale = {location: symbols for location, symbols in stale.items() if symbols}
    require(not stale, f"Drop the entries I1 has already removed: {stale}")
    remaining = sum(len(symbols) for symbols in baseline.values())
    print(f"Speculation: common layers name no new algorithm ({remaining} listed for I1 to remove)")


def _symbols_in(path, symbols):
    """Which of `symbols` still appear in code in `path`."""
    found = set()
    if not path.is_file():
        return found
    for line in magic_masked_lines(path.read_text().splitlines()):
        code = line.split("//", 1)[0]
        for symbol in symbols:
            if ALGORITHM_NAMES.search(code) and symbol in code:
                found.add(symbol)
    return found


PRODUCTION_ROOTS = ("crates", "tools")


def is_test_source(path, root):
    """Whether a file is test source, where unwrapping a failed expectation is the point."""
    relative = path.relative_to(root)
    return "tests" in relative.parts or path.name == "tests.rs" or path.name.startswith("test_")


def check_production_unwraps(root=ROOT):
    """Library and tool sources propagate failures; only test sources may unwrap.

    An `unwrap` in production turns a recoverable condition into a panic that takes the process
    with it. The workspace has none left in production code, so this keeps it that way: the scan
    skips test sources and `#[cfg(test)]` blocks, and covers `build.rs` with the crates and tools.
    """
    paths = [path for name in PRODUCTION_ROOTS for path in (root / name).rglob("*.rs")]
    paths.append(root / "build.rs")
    for path in sorted(paths):
        if not path.is_file() or is_test_source(path, root):
            continue
        lines = magic_masked_lines(path.read_text().splitlines())
        for lineno, line in enumerate(lines, 1):
            code = line.split("//", 1)[0]
            if ".unwrap()" in code:
                location = path.relative_to(root)
                raise SystemExit(
                    f"{location}:{lineno}: production code must propagate this failure instead of "
                    f"unwrapping it: `{line.strip()[:80]}`"
                )
    print("Production: no unwrap outside test sources")


def check_typed_api_json():
    """API responses are typed; dynamic JSON is limited to annotated wire inputs.

    Building response objects with `json!` allocates an intermediate Value tree and loses the
    compile-time shape, which is both slower and easier to drift. The gate also keeps
    `serde_json::Value` out of response and stream code; the allow-list above is only for
    protocol-defined opaque input.
    """
    require(API_ROOT.is_dir(), f"API adapter root is missing: {API_ROOT}")
    for path in sorted(API_ROOT.rglob("*.rs")):
        relative = path.relative_to(API_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("//", 1)[0]
            if re.search(r"\bjson!\s*\(", code):
                raise SystemExit(
                    f"{path.relative_to(ROOT)}:{lineno}: API code must use typed serialization, "
                    "not the json! macro"
                )
            if relative not in API_VALUE_ALLOWED and re.search(r"\bValue\b", code):
                raise SystemExit(
                    f"{path.relative_to(ROOT)}:{lineno}: API response/event code must use typed "
                    "structs, not serde_json::Value (allowed only in wire request DTOs and "
                    "opaque protocol inputs listed in tools/check/policy.py)"
                )
    print("API: typed JSON only; no json! and no response-path Value")


def check_production_dependencies():
    tree = subprocess.run(
        [
            "cargo",
            "tree",
            "--locked",
            "-p",
            "infer-cli",
            "--no-default-features",
            "--edges",
            "normal",
            "--prefix",
            "none",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(
        not any(name in tree for name in ("infer-backend-host", "infer-backend-reference")),
        "Production CLI must not depend on CPU testing executors",
    )


if __name__ == "__main__":
    check_manifests()
    check_exceptions()
    check_magic_numbers()
    check_strategy_neutrality()
    check_algorithm_neutrality()
    check_production_unwraps()
    check_typed_api_json()
    check_production_dependencies()
    print(
        "Lint inheritance, exception, magic number, strategy neutrality, typed API JSON, "
        "and GPU-only dependency checks passed"
    )
