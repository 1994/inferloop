#!/usr/bin/env python3
"""Explicit release-only full quality and frozen multi-pair performance acceptance."""

import argparse
import importlib.util
import json
from pathlib import Path


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def run(args):
    quality = module("release_quality", "run-task-quality.py")
    gate = module("release_pairs", "process-pair-gate.py")
    contract = module("release_contract", "experiment-checklist.py")
    digest = quality.harness.sha256_file(args.executable)
    performance = []
    # Reject incomplete/failed formal evidence before starting another GPU server.
    for checklist_path, pairs_path in args.performance:
        checklist = contract.load(checklist_path)
        pairs = json.loads(pairs_path.read_text())
        if pairs.get("schema") != 1 or not isinstance(pairs.get("pairs"), list):
            raise ValueError("invalid release pairs manifest")

        def load(name, root=pairs_path.parent):
            return json.loads((root / name).read_text())

        for pair in pairs["pairs"]:
            if load(pair["candidate"]).get("binary_sha256") != digest:
                raise ValueError("release performance used a different candidate executable")
        result = gate.evaluate(
            checklist, quality.harness.sha256_file(checklist_path), pairs["pairs"], load
        )
        performance.append(result)
        if not result["passed"]:
            raise ValueError("release performance not demonstrated; quality was not started")
    result = quality.run(args)
    accepted = result.get("quality_accepted") is True
    report = {
        "tier": "release",
        "release_accepted": accepted,
        "performance": performance,
        "quality": result,
    }
    (args.output / "release-gate.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--performance",
        nargs=2,
        type=Path,
        action="append",
        required=True,
        metavar=("CHECKLIST", "PAIRS"),
    )
    for name in ["suite", "model", "executable", "baseline", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--startup-timeout", type=float, default=900)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["release_accepted"] else 1)


if __name__ == "__main__":
    main()
