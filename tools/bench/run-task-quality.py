#!/usr/bin/env python3
"""Collect one complete native task suite per server; reuse an explicit frozen baseline.

This is task-quality evidence, not a release, kernel-numerics or performance certificate.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import itertools
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


harness = module("quality_serving", "serve-compare.py")
scorer = module("quality_scoring", "score-task-quality.py")

# Only rejection is allowed at these looks; complete-suite acceptance stays unchanged.
SCREEN_SIZES = (16, 64, 256)


def document(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def generation(suite):
    settings = suite.get("generation", {})
    defaults = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "enable_thinking": False,
    }
    if any(settings.get(key, value) != value for key, value in defaults.items()):
        raise ValueError("collector supports only the declared greedy generation contract")
    count = settings.get("max_new_tokens", 512)
    if type(count) is not int or count <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    return count


def validate_baseline(suite, report, suite_sha, model, hardware):
    # Recompute raw answers and require the entire aligned suite; never trust a cached pass flag.
    scored = scorer.score(suite, report, suite_sha)
    for key, value in (
        ("model", model),
        ("hardware", hardware),
        ("numeric_contract", suite["numeric_contract"]),
    ):
        if not value or report.get(key) != value:
            raise ValueError(f"baseline identity differs: {key}")
    digest = report.get("binary_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("baseline needs its frozen executable SHA256")
    try:
        bytes.fromhex(digest)
    except ValueError as error:
        raise ValueError("invalid baseline executable SHA256") from error
    return scored


def prefix_count(args, base):
    data = harness.read_metrics(args, base)
    metrics = data.get("metrics") if isinstance(data, dict) else None
    count = metrics.get("prefix_tokens_reused") if isinstance(metrics, dict) else None
    if type(count) is not int or not 0 <= count < 2**64 - 1:
        raise ValueError("native prefix counter is missing, invalid or saturated")
    return count


def stop(process):
    code = process.poll()
    if code is not None:
        raise RuntimeError(f"server exited before shutdown: {code}")
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError as error:
        process.wait(timeout=10)
        raise RuntimeError("server exited before shutdown") from error
    try:
        code = process.wait(timeout=30)
    except subprocess.TimeoutExpired as error:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
        raise RuntimeError("server did not stop gracefully") from error
    if code != 0:
        raise RuntimeError(f"server shutdown failed: {code}")


def result_row(sample, response, decode):
    if response.get("truncated") is not False or not response.get("finished"):
        raise ValueError(f"unfinished task: {sample['id']}")
    if response.get("finish_reason") not in scorer.FINISH_REASONS:
        raise ValueError(f"invalid finish reason: {sample['id']}")
    finished = response["finished"]
    if (
        isinstance(finished, dict)
        and (finished.get("measurement") or {}).get("successful") is not True
    ):
        raise ValueError(f"unsuccessful native task: {sample['id']}")
    ids = response["token_ids"]
    if not ids or any(type(token) is not int or not 0 <= token <= 0xFFFFFFFF for token in ids):
        raise ValueError("task output needs integer token IDs")
    text = decode(ids)
    expected = scorer.numeric_answer(sample["reference_answer"])
    answer = scorer.numeric_answer(text)
    return {
        "id": sample["id"],
        "row_id": sample["row_id"],
        "variant": sample["variant"],
        "input_token_ids": sample["tokens"],
        "token_ids": ids,
        "text": text,
        "correct": answer is not None and answer == expected,
        "finished": response["finished"],
        "finish_reason": response["finish_reason"],
        "truncated": False,
        "prefix_tokens_reused": 0,
    }


def screening_sizes(suite):
    return [size for size in SCREEN_SIZES if size < len(suite["samples"])]


def batch_ranges(count, concurrency, checkpoints):
    start = 0
    while start < count:
        boundary = next((size for size in checkpoints if size > start), count)
        end = min(start + concurrency, boundary, count)
        yield start, end
        start = end


def screen_progress(suite, baseline, rows):
    """Reject clear paired regressions; lack of a small-sample pass is never a rejection.

    Invert the existing score bound with the pair direction reversed to obtain an upper
    bound. Bonferroni allocates the declared error budget across all fixed looks and the
    overall/variant checks. Source-question clustering is retained. This is a diagnostic
    rejection screen, not a complete-suite quality certificate.
    """
    if len(rows) not in screening_sizes(suite):
        return None
    variants = sorted({sample["variant"] for sample in suite["samples"]})
    checks = len(screening_sizes(suite)) * (1 + len(variants))
    confidence = 1 - (1 - suite["acceptance"]["confidence"]) / checks
    margin = suite["acceptance"]["noninferiority_margin_accuracy"]
    pairs = list(zip(baseline[: len(rows)], rows, strict=True))
    groups = [("overall", pairs)] + [
        (variant, [(a, b) for a, b in pairs if a["variant"] == variant]) for variant in variants
    ]
    results = []
    for name, subset in groups:
        if not subset:
            continue
        gains = sum(not a["correct"] and b["correct"] for a, b in subset)
        losses = sum(a["correct"] and not b["correct"] for a, b in subset)
        effect = scorer.design_effect(scorer.paired_changes(subset))
        upper = -scorer.accuracy_change_lower_bound(losses, gains, len(subset), confidence, effect)
        results.append(
            {
                "group": name,
                "tasks": len(subset),
                "gains": gains,
                "losses": losses,
                "accuracy_change": (gains - losses) / len(subset),
                "accuracy_change_upper_bound": upper,
                "design_effect": effect,
                "rejected": upper < -margin,
            }
        )
    return {
        "completed_tasks": len(rows),
        "decision": "reject" if any(row["rejected"] for row in results) else "continue",
        "final_acceptance": False,
        "per_check_confidence": confidence,
        "planned_checks": checks,
        "noninferiority_margin_accuracy": margin,
        "groups": results,
    }


def collect_batches(args, suite, samples, base, report, decode, baseline=None):
    counter = itertools.count(800000)
    with (args.output / "results.jsonl").open("x") as journal:
        checkpoints = screening_sizes(suite) if baseline is not None else []
        for start, end in batch_ranges(len(samples), args.concurrency, checkpoints):
            batch = samples[start:end]
            batch_start = time.monotonic()
            group = [
                {"case": row["id"], "repeat": 0, "slot": slot, "tokens": row["tokens"]}
                for slot, row in enumerate(batch)
            ]
            before = prefix_count(args, base)
            timing = harness.measure_group(args, base, suite["eos_tokens"], counter, group)
            after = prefix_count(args, base)
            if after != before:
                raise ValueError(f"batch {start} prefix counter changed: {before} -> {after}")
            rows = [
                result_row(sample, response, decode)
                for sample, response in zip(batch, timing["results"], strict=True)
            ]
            if any(len(row["token_ids"]) > args.tokens for row in rows):
                raise ValueError("task exceeded its declared generation budget")
            # Only complete, verified batches are committed. The journal survives later failures.
            for row in rows:
                journal.write(json.dumps(row) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            report["results"].extend(rows)
            report["batches"].append(
                {
                    "start": start,
                    "count": len(rows),
                    "wall_seconds": time.monotonic() - batch_start,
                    "prefix_counter_before": before,
                    "prefix_counter_after": after,
                }
            )
            progress = {"completed_tasks": len(report["results"]), "total_tasks": len(samples)}
            (args.output / "progress.json").write_text(json.dumps(progress) + "\n")
            print(json.dumps(progress), flush=True)
            if baseline is not None:
                screen = screen_progress(suite, baseline, report["results"])
                if screen is not None:
                    report["screening"].append(screen)
                    (args.output / "screening.json").write_text(
                        json.dumps(report["screening"], indent=2) + "\n"
                    )
                    print(json.dumps({"quality_screen": screen}), flush=True)
                    if screen["decision"] == "reject":
                        raise RuntimeError(
                            f"early quality regression at {len(report['results'])} tasks"
                        )


def collect(args, suite, suite_sha, model, hardware, decode, baseline=None):
    port = harness.free_port()
    base = f"http://127.0.0.1:{port}"
    command = [str(args.executable), str(args.model), "--listen", f"127.0.0.1:{port}"]
    binary = harness.binary_identity(args.executable)
    if binary.get("release_like") is not True:
        raise ValueError("task collection requires a release-like native executable")
    report = {
        "schema": 1,
        "completed": False,
        "profile_id": suite["profile_id"],
        "suite_sha256": suite_sha,
        "model": model,
        "hardware": hardware,
        "numeric_contract": suite["numeric_contract"],
        "binary_sha256": binary["sha256"],
        "binary": binary,
        "collector_source": harness.git_identity(),
        "command": command,
        "collection": {"concurrency": args.concurrency, "tokens": args.tokens},
        "results": [],
        "batches": [],
        "screening": [],
        "screening_policy": {
            "checkpoints": screening_sizes(suite) if baseline is not None else [],
            "scope": "rejection only; all planned tasks required for acceptance",
            "method": "reversed paired Tango upper bound, Rao-Scott clusters, Bonferroni",
        },
    }
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    log_path = args.output / "server.log"
    with log_path.open("x") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
        try:
            report["startup_seconds"] = harness.wait_ready(
                args, process, base, log_path, args.startup_timeout
            )
            report["initial_readback"] = harness.runtime_readback(args, base, log_path)
            collect_batches(args, suite, suite["samples"], base, report, decode, baseline)
            report["final_readback"] = harness.runtime_readback(args, base, log_path)
            if harness.sha256_file(args.executable) != binary["sha256"]:
                raise ValueError("executable changed during collection")
            report["completed"] = True
        except Exception as error:
            report["error"] = repr(error)
            raise
        finally:
            try:
                stop(process)
            except Exception as error:
                report["completed"] = False
                report["error"] = repr(error)
                raise
            finally:
                report["count"] = len(report["results"])
                report["correct"] = sum(row["correct"] for row in report["results"])
                (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    scorer.score(suite, report, suite_sha)
    return report


def run(args, decode=None):
    if args.concurrency < 1 or args.startup_timeout <= 0:
        raise ValueError("concurrency and startup timeout must be positive")
    suite, suite_sha = document(args.suite)
    scorer.validate_suite(suite)
    args.tokens = generation(suite)
    args.engine = "native"
    harness.require_protected_run()
    model, hardware = harness.model_identity(args.model), harness.hardware_identity()
    if not hardware.get("devices"):
        raise ValueError("task collection requires a visible CUDA device identity")
    for name, digest in suite.get("tokenizer_files", {}).items():
        if model["files"].get(name) != digest:
            raise ValueError(f"suite tokenizer artifact differs: {name}")
    if not suite.get("eos_tokens") or set(suite["eos_tokens"]) != set(
        harness.eos_tokens(args.model)
    ):
        raise ValueError("suite EOS tokens differ from the model")
    baseline = baseline_sha = baseline_rows = None
    if args.baseline:
        baseline, baseline_sha = document(args.baseline)
        baseline_rows = validate_baseline(suite, baseline, suite_sha, model, hardware)
        previous = baseline.get("collection")
        if previous and previous != {"concurrency": args.concurrency, "tokens": args.tokens}:
            raise ValueError("baseline collection settings differ")
    if args.output.exists():
        raise ValueError("output already exists; use a new evidence directory")
    if decode is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

        def decode(ids):
            return tokenizer.decode(ids, skip_special_tokens=True)

    args.output.mkdir(parents=True)
    report = collect(args, suite, suite_sha, model, hardware, decode, baseline_rows)
    if baseline is None:
        return {"collection_status": "complete", "quality_status": "not_evaluated"}
    if harness.sha256_file(args.baseline) != baseline_sha:
        raise ValueError("frozen baseline changed during collection")
    result = scorer.compare(suite, baseline, report, suite_sha)
    result["evidence"] = {
        "suite_sha256": suite_sha,
        "baseline_sha256": baseline_sha,
        "candidate_sha256": harness.sha256_file(args.output / "report.json"),
        "baseline_reused": True,
    }
    (args.output / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, help="explicit complete frozen quality report")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--startup-timeout", type=float, default=900)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        parser.exit(1, f"task collection failed: {error}\n")
    print(json.dumps(result, indent=2))
    if args.baseline and not result["quality_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
