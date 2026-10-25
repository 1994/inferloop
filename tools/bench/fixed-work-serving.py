#!/usr/bin/env python3
"""Fixed-output serving timing for one fresh server process.

Every request generates exactly `--tokens` tokens on both engines (EOS ignored), so builds and
engines do equal work. The first group of `--inputs` is an excluded warmup; each later group is
released together and timed as one makespan. Token digests let two builds prove identical
outputs. Server flags after `--` are passed through; none means the engine's own defaults.

Run inside `tools/bench/safe-run.sh`, one process per arm, alternating arms (ABBA...).
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import signal
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

HARNESS = Path(__file__).with_name("serve-compare.py")
PREFILL_PROFILE_ENV = "INFER_CUDA_PREFILL_PROFILE"
EXECUTION_PROFILE_ENV = "INFER_CUDA_EXECUTION_PROFILE"


def harness():
    spec = importlib.util.spec_from_file_location("serve_compare", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def server_command(engine, executable, model, port, extra):
    if engine == "native":
        return [str(executable), str(model), "--listen", f"127.0.0.1:{port}", *extra]
    return [
        str(executable),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *extra,
    ]


def fixed_work(payload):
    """OpenAI completions stop at the model EOS unless told otherwise; native gets no EOS."""
    if isinstance(payload, dict) and "prompt" in payload and "max_tokens" in payload:
        return {**payload, "ignore_eos": True}
    return payload


def groups(rows):
    """Request groups in `repeat` order; each row needs `case`, `repeat` and `tokens`."""
    ordered = sorted({row["repeat"] for row in rows})
    return [
        [
            {"case": row["case"], "repeat": row["repeat"], "slot": slot, "tokens": row["tokens"]}
            for slot, row in enumerate(r for r in rows if r["repeat"] == repeat)
        ]
        for repeat in ordered
    ]


def digest(token_ids):
    return hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()


def summary(report):
    measured = [group["wall_seconds"] for group in report["groups"] if not group["warmup"]]
    return {
        "completed": report["completed"],
        "measured_walls": measured,
        "median_wall": statistics.median(measured) if measured else None,
    }


def environment(engine, h, profile_stem):
    if engine != "native":
        return h.process_environment(engine)
    # Diagnostic switches must not leak in from the caller; only `--profile` sets them.
    env = {k: v for k, v in os.environ.items() if not k.startswith("INFER_")}
    if profile_stem is not None:
        env[PREFILL_PROFILE_ENV] = f"{profile_stem}.events.jsonl"
        env[EXECUTION_PROFILE_ENV] = f"{profile_stem}.execution.jsonl"
    return env


def run(args, h):
    dumps = h.json.dumps
    h.json.dumps = lambda value, *a, **k: dumps(fixed_work(value), *a, **k)
    rows = json.loads(args.inputs.read_text())
    port = h.free_port()
    base = f"http://127.0.0.1:{port}"
    command = server_command(args.engine, args.executable, args.model, port, args.extra)
    stem = args.output.with_suffix("")
    env = environment(args.engine, h, stem if args.profile else None)
    probe = SimpleNamespace(
        engine=args.engine, executable=args.executable, model=args.model, tokens=args.tokens
    )
    report = {"engine": args.engine, "command": command, "completed": False, "groups": []}
    report["binary_sha256"] = h.sha256_file(args.executable)
    log_path = Path(f"{stem}.server.log")
    with log_path.open("w") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            h.wait_ready(probe, child, base, log_path, args.startup_timeout)
            counter = itertools.count(900000)
            for index, group in enumerate(groups(rows)):
                timing = h.measure_group(probe, base, [], counter, group)
                outputs = [len(result["token_ids"]) for result in timing["results"]]
                if any(count != args.tokens for count in outputs):
                    raise RuntimeError(f"fixed work violated: {outputs}")
                requests = [
                    {
                        **{
                            k: v
                            for k, v in result.items()
                            if k not in ("token_ids", "events", "finished")
                        },
                        "token_digest": digest(result["token_ids"]),
                    }
                    for result in timing["results"]
                ]
                report["groups"].append(
                    {
                        "warmup": index == 0,
                        "wall_seconds": timing["wall_seconds"],
                        "requests": requests,
                    }
                )
                print(json.dumps({"group": index, "wall": timing["wall_seconds"]}), flush=True)
            report["completed"] = True
        except Exception as error:
            report["error"] = repr(error)
            raise
        finally:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            args.output.write_text(json.dumps(report, indent=1))
    print(json.dumps(summary(report)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=["native", "vllm"])
    parser.add_argument("--executable", required=True, type=Path, help="infer or vLLM python")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--inputs", required=True, type=Path, help="JSON rows: case, repeat, tokens"
    )
    parser.add_argument("--output", required=True, type=Path, help="new report path (.json)")
    parser.add_argument("--tokens", type=int, default=256, help="generated tokens per request")
    parser.add_argument("--startup-timeout", type=float, default=1500.0)
    parser.add_argument(
        "--profile", action="store_true", help="native only: per-node graph events (not timing)"
    )
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="-- then server flags")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new evidence path")
    if args.profile and args.engine != "native":
        parser.error("--profile applies to the native engine only")
    args.extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    run(args, harness())


if __name__ == "__main__":
    sys.exit(main())
