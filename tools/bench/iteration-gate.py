#!/usr/bin/env python3
"""One native process pair and 64 quality tasks; never a release certificate."""

import argparse
import copy
import importlib.util
import itertools
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "task_quality", Path(__file__).with_name("run-task-quality.py")
)
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)
harness = quality.harness
TASKS = 64
BUDGET_SECONDS = 600
WORK_SECONDS = 580


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def remaining(args):
    seconds = args.deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("iteration budget exhausted; no acceptance issued")
    return seconds


def metrics(trial):
    rows = trial["results"]
    return {
        "wall_seconds": trial["wall_seconds"],
        "ttft_seconds": statistics.median(row["ttft_seconds"] for row in rows),
        "tpot_seconds": statistics.median(row["tpot_seconds"] for row in rows),
    }


def performance(args, executable, label):
    remaining(args)
    dest = args.output / label
    dest.mkdir()
    tuning = dest / "tuning.json"
    tuning.write_bytes(args.tuning_seed.read_bytes())
    env = os.environ.copy()
    env["INFER_CUDA_TUNING_TABLE"] = str(tuning)
    native = copy.copy(args)
    native.engine = "native"
    port = harness.free_port()
    base = f"http://127.0.0.1:{port}"
    command = [str(executable), str(args.model), "--listen", f"127.0.0.1:{port}"]
    report = {"completed": False, "command": command, "trials": [], "metrics": {}}
    counter = itertools.count(1)
    with (dest / "server.log").open("x") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
        try:
            report["startup_seconds"] = harness.wait_ready(
                native,
                process,
                base,
                dest / "server.log",
                min(args.startup_timeout, remaining(args)),
            )
            report["readback"] = harness.runtime_readback(native, base, dest / "server.log")
            for name, path in [("c16", args.workload), ("short1", args.short_workload)]:
                rows = harness.load_workload(path, 1)
                for repeat in [-1, 0]:
                    remaining(args)
                    group = [row for row in rows if row["repeat"] == repeat]
                    if not group:
                        raise ValueError(f"{name} has no warmup/measured workload")
                    before = quality.prefix_count(native, base)
                    trial = harness.measure_group(native, base, [], counter, group)
                    after = quality.prefix_count(native, base)
                    if before != 0 or after != 0:
                        raise ValueError("performance request reused a prefix")
                    if len(trial["results"]) != len(group):
                        raise ValueError("performance response count differs")
                    for row in trial["results"]:
                        if (
                            row.get("truncated") is not False
                            or len(row["token_ids"]) != args.tokens
                            or not row.get("tpot_available")
                            or (row.get("finished") or {}).get("measurement", {}).get("successful")
                            is not True
                        ):
                            raise ValueError("incomplete or unsuccessful performance response")
                    trial.update(name=name, repeat=repeat)
                    report["trials"].append(trial)
                    if repeat == 0:
                        report["metrics"][name] = metrics(trial)
                        print(
                            json.dumps({"phase": label, "case": name, **report["metrics"][name]}),
                            flush=True,
                        )
                    write(dest / "report.json", report)
            report["completed"] = True
        finally:
            try:
                quality.stop(process)
            except BaseException:
                report["completed"] = False
                raise
            finally:
                write(dest / "report.json", report)
    remaining(args)
    return report


def quality_summary(report, baseline_rows):
    rows = report["results"]
    count = len(rows)
    correct = sum(row["correct"] for row in rows)
    reference = sum(row["correct"] for row in baseline_rows[:count])
    return {
        "tasks": count,
        "correct": correct,
        "baseline_correct": reference,
        "accuracy": correct / count if count else None,
        "accuracy_change_percentage_points": 100 * (correct - reference) / count if count else None,
        "screening": report["screening"],
        "release_quality_accepted": False,
    }


def compare(left, right, limit):
    a = left["readback"]["readback"]
    b = right["readback"]["readback"]
    if not a.get("execution_profile") or a.get("execution_profile") != b.get("execution_profile"):
        raise ValueError("runtime execution profiles differ")
    keys = ("width", "capacity", "kv_tokens", "prompt_width", "envelope_bytes")
    plans = [row.get("serving_plan") for row in (a, b)]
    if any(not plan or any(key not in plan for key in keys) for plan in plans):
        raise ValueError("missing funded serving resources")
    if any(plans[0][key] != plans[1][key] for key in keys):
        raise ValueError("funded serving resources differ; declare a separate release profile")
    for plan in plans:
        if max(plan["quoted_bytes"], plan["allocated_bytes"]) > plan["envelope_bytes"]:
            raise ValueError("serving resources exceed budget")
    result = {}
    for name, baseline in left["metrics"].items():
        candidate = right["metrics"][name]
        changes = {key: 100 * (candidate[key] / value - 1) for key, value in baseline.items()}
        result[name] = {
            "baseline": baseline,
            "candidate": candidate,
            "change_percent": changes,
            "gross_regression": any(value > limit for value in changes.values()),
        }
    return result


def run(args, decode=None):
    started = time.monotonic()
    args.deadline = started + WORK_SECONDS
    if args.output.exists():
        raise ValueError("output already exists; use a new evidence directory")
    harness.require_protected_run()
    suite, suite_sha = quality.document(args.suite)
    quality.scorer.validate_suite(suite)
    if len(suite["samples"]) <= TASKS:
        raise ValueError("iteration screening needs an original suite larger than 64 tasks")
    baseline, _baseline_sha = quality.document(args.baseline)
    model, hardware = harness.model_identity(args.model), harness.hardware_identity()
    if not hardware.get("devices"):
        raise ValueError("no visible CUDA device")
    scored = quality.validate_baseline(suite, baseline, suite_sha, model, hardware)
    tokens = quality.generation(suite)
    if set(suite["eos_tokens"]) != set(harness.eos_tokens(args.model)):
        raise ValueError("suite EOS differs from model")
    for name, digest in suite.get("tokenizer_files", {}).items():
        if model["files"].get(name) != digest:
            raise ValueError(f"tokenizer differs: {name}")
    expected = {"concurrency": args.concurrency, "tokens": tokens}
    if baseline.get("collection") and baseline["collection"] != expected:
        raise ValueError("baseline collection settings differ")
    identities = {
        name: harness.binary_identity(path)
        for name, path in [("baseline", args.baseline_executable), ("candidate", args.executable)]
    }
    if any(value.get("release_like") is not True for value in identities.values()):
        raise ValueError("both executables must be release-like")
    hashes = {
        str(path): harness.sha256_file(path)
        for path in [
            args.suite,
            args.baseline,
            args.workload,
            args.short_workload,
            args.tuning_seed,
            args.baseline_executable,
            args.executable,
        ]
    }
    args.output.mkdir(parents=True)
    summary = {
        "tier": "iteration",
        "completed": False,
        "iteration_accepted": False,
        "release_accepted": False,
        "performance_accepted": False,
        "quality_accepted": False,
        "time_budget_seconds": BUDGET_SECONDS,
        "sample_process_pairs": 1,
        "quality_tasks": TASKS,
        "identity": identities,
        "model": model,
        "hardware": hardware,
        "inputs_sha256": hashes,
        "gross_regression_limit_percent": args.regression_percent,
        "scope": "one-pair performance snapshot and rejection-only quality screen",
    }
    try:
        summary["phase"] = "baseline performance"
        write(args.output / "summary.json", summary)
        left = performance(args, args.baseline_executable, "baseline-performance")
        summary["phase"] = "candidate performance"
        write(args.output / "summary.json", summary)
        right = performance(args, args.executable, "candidate-performance")
        summary["performance"] = compare(left, right, args.regression_percent)
        summary["phase"] = "64-task quality screen"
        write(args.output / "summary.json", summary)
        if any(row["gross_regression"] for row in summary["performance"].values()):
            raise ValueError("gross performance regression; stop before quality collection")
        if decode is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

            def decode(ids):
                return tokenizer.decode(ids, skip_special_tokens=True)

        qa = SimpleNamespace(**vars(args))
        qa.engine, qa.tokens, qa.output = "native", tokens, args.output / "quality"
        qa.startup_timeout = min(args.startup_timeout, remaining(args))
        qa.output.mkdir()
        tuning = qa.output / "tuning.json"
        tuning.write_bytes(args.tuning_seed.read_bytes())
        previous = os.environ.get("INFER_CUDA_TUNING_TABLE")
        os.environ["INFER_CUDA_TUNING_TABLE"] = str(tuning)
        try:
            report = quality.collect(
                qa, suite, suite_sha, model, hardware, decode, scored, sample_limit=TASKS
            )
        finally:
            if previous is None:
                os.environ.pop("INFER_CUDA_TUNING_TABLE", None)
            else:
                os.environ["INFER_CUDA_TUNING_TABLE"] = previous
        summary["quality"] = quality_summary(report, scored)
        if not report.get("collection_completed") or len(report["results"]) != TASKS:
            raise ValueError("incomplete iteration quality screen")
        for path, digest in hashes.items():
            if harness.sha256_file(Path(path)) != digest:
                raise ValueError(f"frozen input changed: {path}")
        remaining(args)
        summary.update(completed=True, iteration_accepted=True, phase="complete")
    except BaseException as error:
        summary["error"] = repr(error)
        path = args.output / "quality" / "report.json"
        if path.exists():
            summary["quality"] = quality_summary(json.loads(path.read_text()), scored)
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        write(args.output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "model",
        "executable",
        "baseline-executable",
        "suite",
        "baseline",
        "workload",
        "short-workload",
        "tuning-seed",
        "output",
    ]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("--regression-percent", type=float, default=10)
    args = parser.parse_args()
    if min(args.tokens, args.concurrency, args.startup_timeout) <= 0 or args.regression_percent < 0:
        parser.error("budgets/concurrency must be positive; regression limit must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
