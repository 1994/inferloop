#!/usr/bin/env python3
"""Inventory the Rust test corpus and hold the collection ratchet.

Private unit tests live next to the module they test (`src/<module>/tests.rs` or
`src/<module>/<name>.rs`) and are mounted with a plain `#[cfg(test)] mod ...;`; integration tests
live in each crate's `tests/`. Test bodies are not mounted through `#[path]`; cross-crate fixtures
and acceptance cases live in the `crates/testing/` support crates instead. The plan also removes
the two CPU test executors; the inventory records which files still consume them.

This tool records the corpus in `tools/check/test-inventory.json` and enforces the invariants:

* every `#[path = ...]` mount must resolve and be mounted exactly once;
* every `tests.rs`/`*_tests.rs`/`*_check.rs` test file under `src/` must be mounted by a module,
  or its cases never run;
* no new consumer of the CPU test executors or the `test-backends` feature may appear;
* a change in the collected total must be recorded with `--record`, so a migration that loses
  assertions cannot be mistaken for progress.

Run with `--record` to accept the current tree as the new baseline after a deliberate migration.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(__file__).with_name("test-inventory.json")
# A test entry, however it is spelled in this workspace.
TEST_ENTRY = re.compile(r"#\[(?:tokio::)?test\]")
# `#[path = "....rs"]` mounts a shared test source that cannot live inside one module.
PATH_MOUNT = re.compile(r'#\[\s*path\s*=\s*"([^"]+)"\s*\]')
# `[[test]] name = "..."` declares an integration target by hand.
TEST_TARGET = re.compile(r'\[\[test\]\][^\[]*?name\s*=\s*"([^"]+)"', re.DOTALL)
# Gating attributes that decide whether a case runs by default.
GATE = re.compile(
    r"#\[(?:cfg\(([^)]*)\)|ignore\s*=\s*\"([^\"]*)\")[^\]]*\]",
    re.MULTILINE,
)
EXECUTOR_PATTERNS = (
    "infer-backend-host",
    "infer_backend_host",
    "infer-backend-reference",
    "infer_backend_reference",
    "test-backends",
    "test_backends",
)
# The migration's own tooling names these on purpose; only real consumers are recorded.
EXECUTOR_SCAN_EXCLUDED = (
    "tools/check/test-inventory.py",
    "tools/check/test_test_inventory.py",
)
# Vendored environments and local acceptance output are not project sources, the same way
# `.gitignore` and the credential gate treat them.
SCAN_EXCLUDED_PREFIXES = ("artifacts/", "target/", "dist/")
SCAN_EXCLUDED_PARTS = ("site-packages", "node_modules")

# What each consumer needs the executor for, which decides where its assertions go when the
# executors are deleted. `plumbing` is build and feature wiring rather than a scene.
ROLE_RULES = (
    (r"^crates/testing/cpu/", "fixture"),
    (r"^crates/foundation/ir/", "plumbing"),
    (r"^crates/backend/kernel-api/Cargo\.toml$", "plumbing"),
    (r"^crates/backend/cuda/Cargo\.toml$", "plumbing"),
    (r"^tools/validation/", "protocol"),
    (r"^tools/check/", "plumbing"),
    (r"^tools/package/", "plumbing"),
    (r"^Cargo\.toml$", "plumbing"),
    (r"^tools/bench/cpu/", "benchmark"),
    (r"^crates/backend/cuda/examples/", "numeric"),
    (r"^crates/engine/runtime/", "protocol"),
    (r"^crates/service/frontdoor/", "protocol"),
    (r"^crates/service/agent/", "protocol"),
    (r"^crates/service/cli/", "service"),
)


def role(path):
    """The removal plan's classification for one consumer, or None if it needs a decision."""
    for pattern, name in ROLE_RULES:
        if re.search(pattern, path):
            return name
    return None


def crates():
    return sorted(path for path in REPO_ROOT.glob("crates/*/*") if (path / "Cargo.toml").is_file())


def classify(relative):
    """Which layout a test file belongs to, in the plan's terms."""
    if relative.startswith("tests/"):
        return "tests"
    if relative.startswith("benches/"):
        return "benches"
    name = Path(relative).name
    if (
        "/tests/" in f"/{relative}"
        or name.endswith("_tests.rs")
        or name
        in (
            "tests.rs",
            "test.rs",
            "bench_check.rs",
        )
    ):
        return "src_near"
    return "src_inline"


def uncommitted():
    """Repo-relative paths whose working-tree content is not what the index holds.

    Recording the corpus from a tree that still holds someone else's uncommitted tests was how a
    record came to describe 475 cases when the committed tree held 467. The record has to describe
    what is committed, so files that differ from the index are left out and reported.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    return dirty_paths(result.stdout)


def dirty_paths(porcelain):
    """Paths whose working-tree content is not what the index holds.

    Porcelain's two status columns are the index and the worktree. A staged change (`M `) is what
    will be committed and counts; an edit after staging (`MM`), an unstaged edit (` M`) or an
    untracked file (`??`) does not, because the record must describe the branch rather than the
    checkout. Renames resolve to the new name.
    """
    paths = set()
    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        index, worktree = line[0], line[1]
        if index == "?" or worktree != " ":
            name = line[3:].strip()
            if " -> " in name:
                name = name.split(" -> ", 1)[1]
            paths.add(name.strip('"'))
    return paths


def scan_crate(root, dirty=frozenset()):
    entries = {"src_inline": 0, "src_near": 0, "tests": 0, "benches": 0}
    ignored = {}
    gated = {}
    mounts = []
    for directory in ("src", "tests", "benches"):
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.rs")):
            # Tests may scan a temporary crate outside the repository, which has no status.
            try:
                relative_repo = str(path.relative_to(REPO_ROOT))
            except ValueError:
                relative_repo = None
            if relative_repo is not None and relative_repo in dirty:
                scan_crate.skipped.add(relative_repo)
                continue
            text = path.read_text(errors="ignore")
            found = len(TEST_ENTRY.findall(text))
            relative = str(path.relative_to(root))
            if found:
                entries[classify(relative)] += found
                for match in GATE.finditer(text):
                    condition, reason = match.group(1), match.group(2)
                    if reason is not None:
                        ignored[reason] = ignored.get(reason, 0) + 1
                    elif condition and ("target_os" in condition or "feature" in condition):
                        gated[condition] = gated.get(condition, 0) + 1
            mounts.extend(
                {"crate": root.name, "file": relative, "mount": mount}
                for mount in PATH_MOUNT.findall(text)
            )
    return entries, ignored, gated, mounts


scan_crate.skipped = set()


def collect():
    dirty = uncommitted()
    scan_crate.skipped = set()
    inventory = {}
    mounts = []
    ignored = {}
    gated = {}
    for root in crates():
        entries, crate_ignored, crate_gated, crate_mounts = scan_crate(root, dirty)
        if sum(entries.values()):
            inventory[root.name] = entries
        for reason, count in crate_ignored.items():
            ignored[reason] = ignored.get(reason, 0) + count
        for condition, count in crate_gated.items():
            gated[condition] = gated.get(condition, 0) + count
        mounts.extend(crate_mounts)
    totals = {
        key: sum(crate.get(key, 0) for crate in inventory.values())
        for key in ("src_inline", "src_near", "tests", "benches")
    }
    totals["src"] = totals["src_inline"] + totals["src_near"]
    totals["all"] = totals["src"] + totals["tests"] + totals["benches"]
    # Consumers are recorded by crate and by which patterns they mention, not by path: the
    # migration moves these files by design, and a relocated consumer is not a new dependency.
    consumers = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in (".rs", ".toml", ".sh", ".py"):
            continue
        relative = str(path.relative_to(REPO_ROOT))
        if relative in EXECUTOR_SCAN_EXCLUDED:
            continue
        if relative.startswith(SCAN_EXCLUDED_PREFIXES) or any(
            part in relative for part in SCAN_EXCLUDED_PARTS
        ):
            continue
        text = path.read_text(errors="ignore")
        patterns = sorted({p for p in EXECUTOR_PATTERNS if p in text})
        if not patterns:
            continue
        parts = Path(relative).parts
        crate = parts[2] if len(parts) > 2 and parts[0] == "crates" else parts[0]
        consumers.append(
            {"path": relative, "crate": crate, "patterns": patterns, "role": role(relative)}
        )
    return {
        "crates": inventory,
        "totals": totals,
        "ignored_reasons": ignored,
        "gated": gated,
        "path_mounts": mounts,
        "executor_consumers": consumers,
    }


def check_mounts(inventory, root=REPO_ROOT):
    """Every mount resolves, and no file is mounted twice."""
    problems = []
    seen = {}
    for mount in inventory["path_mounts"]:
        crate = root / "crates"
        owner = next((path for path in crate.glob(f"*/{mount['crate']}") if path.is_dir()), None)
        key = (mount["crate"], mount["file"], mount["mount"])
        seen[key] = seen.get(key, 0) + 1
        if owner is None:
            problems.append(f"{mount['crate']}: unknown crate")
            continue
        resolved = (owner / mount["file"]).parent / mount["mount"]
        if not resolved.is_file():
            problems.append(f"{mount['crate']}: {mount['file']} mounts missing {mount['mount']}")
    for key, count in seen.items():
        if count > 1:
            problems.append(f"{key[0]}: {key[1]} mounts {key[2]} {count} times")
    return problems


def check_orphans(root=REPO_ROOT):
    """Every test file must be reachable as a module, or its cases never run.

    A file named `tests.rs`, `*_tests.rs` or `*_check.rs` can sit next to the code it tests and be
    silently omitted from the module tree, which no compiler warning reports: the cases simply do
    not exist. The plan asks for the mount path to exist and be unique, and this is the half of
    that rule which a missing declaration would break.
    """
    problems = []
    for crate in sorted(
        path for path in (root / "crates").glob("*/*") if (path / "Cargo.toml").is_file()
    ):
        src = crate / "src"
        for path in sorted(src.rglob("*.rs")):
            name = path.name
            if not (name == "tests.rs" or name.endswith(("_tests.rs", "_check.rs"))):
                continue
            text = path.read_text(errors="ignore")
            if not TEST_ENTRY.search(text):
                continue
            parent, stem = path.parent, path.stem
            candidates = (
                [src / "lib.rs"]
                if parent == src
                else [parent.parent / f"{parent.name}.rs", parent / "mod.rs"]
            )
            mounted = any(
                re.search(rf"(?:pub\s+)?mod\s+{re.escape(stem)}\s*;", candidate.read_text())
                for candidate in candidates
                if candidate.is_file()
            )
            if not mounted:
                wanted = os.path.normpath(path)
                mounted = any(
                    os.path.normpath(candidate.parent / value) == wanted
                    for candidate in src.rglob("*.rs")
                    for value in PATH_MOUNT.findall(candidate.read_text())
                )
            if not mounted:
                problems.append(
                    f"{crate.name}: {path.relative_to(crate)} is a test file no module mounts, so "
                    "its cases never run"
                )
    return problems


def check_targets(root=REPO_ROOT):
    """Every test file must actually be collected, and helper files must not be.

    `autotests = false` is required once case bodies or helpers live under `tests/`, but it also
    switches off discovery of the platform integration tests that live there, so each of those has
    to be declared. Both mistakes are silent: the first removes targets from the run, the second
    turns a helper into its own binary.
    """
    problems = []
    for crate in sorted(
        path for path in (root / "crates").glob("*/*") if (path / "Cargo.toml").is_file()
    ):
        manifest = crate / "Cargo.toml"
        text = manifest.read_text()
        declared = set(TEST_TARGET.findall(text))
        off = re.search(r"^autotests\s*=\s*false", text, re.MULTILINE) is not None
        tests = crate / "tests"
        helpers = (
            [
                path
                for path in list(tests.glob("unit/*.rs")) + list(tests.glob("support/*.rs"))
                if (tests / "unit").is_dir() or (tests / "support").is_dir()
            ]
            if tests.is_dir()
            else []
        )
        if helpers and not off:
            problems.append(
                f"{crate.name}: case bodies under tests/ need `autotests = false`, otherwise "
                "Cargo builds each of them as its own integration binary"
            )
        if off:
            problems.extend(
                f"{crate.name}: tests/{path.name} is not declared as a [[test]] target "
                "while autotests is off, so it is not collected"
                for path in sorted(tests.glob("*.rs"))
                if path.stem not in declared
            )
    return problems


def regressions(inventory, recorded, root=REPO_ROOT):
    """The ratchet: what may not get worse while the migration proceeds."""
    problems = check_mounts(inventory, root) + check_targets(root) + check_orphans(root)
    if inventory["totals"]["all"] < recorded["totals"]["all"]:
        problems.append(
            f"collected test entries fell to {inventory['totals']['all']} from "
            f"{recorded['totals']['all']}; a smaller corpus is not progress - record where each "
            "removed case went, or re-record deliberately"
        )

    def surface(consumers):
        """The dependency surface: which crate depends on which patterns.

        Counting files would flag the migration itself, because moving a block out of a source
        file turns one consumer into two without adding a dependency.
        """
        return {(consumer["crate"], tuple(consumer["patterns"])) for consumer in consumers}

    for consumer in inventory["executor_consumers"]:
        if consumer.get("role") is None:
            problems.append(
                f"{consumer['path']}: consumer of a CPU test executor with no removal role; add "
                "it to ROLE_RULES so the worklist stays complete"
            )
    before, after = (
        surface(recorded["executor_consumers"]),
        surface(inventory["executor_consumers"]),
    )
    for crate, patterns in sorted(after - before):
        examples = [
            consumer["path"]
            for consumer in inventory["executor_consumers"]
            if (consumer["crate"], tuple(consumer["patterns"])) == (crate, patterns)
        ][:3]
        problems.append(
            f"{crate}: new dependency on a CPU test executor via {', '.join(patterns)} "
            f"({', '.join(examples)})"
        )
    return problems


def render(inventory):
    lines = [
        "| crate | src inline | src near-test | tests/ | benches/ |",
        "|---|---:|---:|---:|---:|",
    ]
    for crate, entries in sorted(
        inventory["crates"].items(),
        key=lambda item: -(item[1]["src_inline"] + item[1]["src_near"]),
    ):
        lines.append(
            f"| {crate} | {entries['src_inline']} | {entries['src_near']} | "
            f"{entries['tests']} | {entries['benches']} |"
        )
    totals = inventory["totals"]
    lines.append(
        f"| **total** | **{totals['src_inline']}** | **{totals['src_near']}** | "
        f"**{totals['tests']}** | **{totals['benches']}** |"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="accept the tree as the baseline")
    parser.add_argument("--json", action="store_true", help="print the inventory")
    parser.add_argument("--table", action="store_true", help="print the per-crate table")
    args = parser.parse_args()
    inventory = collect()
    if args.json:
        print(json.dumps(inventory, indent=2, sort_keys=True))
    if args.table:
        print(render(inventory))
    if args.record:
        SNAPSHOT.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
        if scan_crate.skipped:
            print(
                f"skipped {len(scan_crate.skipped)} uncommitted test file(s); stage them to "
                f"include their cases: {sorted(scan_crate.skipped)}",
                file=sys.stderr,
            )
        print(
            f"recorded {inventory['totals']['all']} test entries "
            f"({inventory['totals']['src']} in src) and "
            f"{len(inventory['executor_consumers'])} executor consumers",
            file=sys.stderr,
        )
        return 0
    recorded = json.loads(SNAPSHOT.read_text())
    problems = regressions(inventory, recorded)
    if scan_crate.skipped:
        # A dirty tree legitimately holds fewer cases than the snapshot; say which files were left
        # out rather than reporting a lost corpus, because the loss is the tree's, not the branch's.
        problems = [
            f"{len(scan_crate.skipped)} uncommitted test file(s) are excluded from this count; "
            f"stage or commit them before judging the corpus: {sorted(scan_crate.skipped)}"
        ] + [problem for problem in problems if "fell to" not in problem]
    for problem in problems:
        print(problem, file=sys.stderr)
    if not args.json and not args.table and not problems:
        delta = inventory["totals"]["all"] - recorded["totals"]["all"]
        note = "" if delta == 0 else f"; {delta:+d} since the snapshot"
        print(
            f"{inventory['totals']['all']} test entries "
            f"({inventory['totals']['src']} still in src){note}; ratchet holds"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
