#!/usr/bin/env python3
"""Paired release check for a refactor: a candidate binary against a base binary.

`pair` serves each configuration from both binaries in turn with the same warm tuning cache and
reports, per configuration: the greedy output hash, fixed-work throughput, tuning measurements
taken while serving, and tuning-table hits grouped by message. A refactor that must not change
behaviour passes when every hash matches, nothing is measured and the hits agree.

`pressure` sends more concurrent long greedy requests than the KV pool holds at once, from both
binaries, and requires every request to finish with the same output on both sides: on-demand
page growth and recompute preemption must not change what a request generates.

`gap` serves one binary under closed-loop load and reports the device-busy share per step, the
evidence the P2 depth-2 and on-demand growth decisions are made from.
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PORT = 18081
LOAD_LIMIT_S = 1500
REQUEST_LIMIT_S = 900
GREEDY_TOKENS = 256
LOAD_TOKENS = 128
GAP_TOKENS = 200
WARM_TOKENS = 16
GAP_REQUESTS_PER_LANE = 3
SEED = 7
HASH_CHARS = 12
PRESSURE_METRICS = ("preemptions", "failed_requests", "successful_requests", "deferred_reasons")
SHUTDOWN_S = 30
FILLER = "The quick brown fox jumps over the lazy dog. " * 28
PROMPTS = (
    "Name three primes.",
    "Explain TCP slow start.",
    "Write a haiku about rust.",
    "Sum 17 and 25, step by step.",
    "What is a B-tree?",
    "Translate good morning to French.",
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# A tuning line names its message, then its key fields; the message is what the hit count groups.
TUNING = re.compile(r"infer::tuning: (.*?)(?: (?:shape|key|rows)=.*)?$")
HIT = re.compile(r"table-cache|from_table=true")
BAD = re.compile(r"fallback|invalid|incomplete")
MEASURED = re.compile(r'source="?measured|searched')


def tuning_hits(log):
    """Count tuning-table hits and misses by message in one server log."""
    hits, bad = Counter(), Counter()
    for line in ANSI.sub("", log).splitlines():
        found = TUNING.search(line)
        if not found:
            continue
        if HIT.search(line):
            hits[found.group(1)] += 1
        if BAD.search(line):
            bad[found.group(1)] += 1
    return hits, bad


def measured(log):
    """Count tuning measurements taken while serving; a warm cache takes none."""
    return sum(1 for line in ANSI.sub("", log).splitlines() if MEASURED.search(line))


def verdict(candidate, base):
    """Return the reasons a candidate run differs from its base run; empty means equivalent."""
    reasons = []
    if candidate["greedy_sha"] != base["greedy_sha"]:
        reasons.append("greedy output differs")
    for side, run in (("candidate", candidate), ("base", base)):
        if run["measured"]:
            reasons.append(f"{side} measured {run['measured']} times on a warm cache")
        if run["bad"]:
            reasons.append(f"{side} logged {sum(run['bad'].values())} tuning misses")
        errors = sum(level["errors"] for level in run["throughput"].values())
        if errors:
            reasons.append(f"{side} had {errors} failed requests")
    if set(candidate["hits"]) != set(base["hits"]):
        reasons.append("tuning tables hit differ: " + ", ".join(
            sorted(set(candidate["hits"]) ^ set(base["hits"]))))
    return reasons


class Server:
    """One `infer serve` process with its own tuning cache, stopped on exit."""

    def __init__(self, binary, model, cache, log, extra):
        os.makedirs(cache, exist_ok=True)
        self.url = f"http://127.0.0.1:{PORT}"
        self.log = open(log, "w")
        self.process = subprocess.Popen(
            [binary, "serve", "--package", model, "--listen", f"127.0.0.1:{PORT}", *extra],
            env=dict(os.environ, XDG_CACHE_HOME=str(cache)),
            stdout=self.log,
            stderr=self.log,
        )

    def __enter__(self):
        deadline = time.monotonic() + LOAD_LIMIT_S
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited with {self.process.returncode}")
            try:
                self.model = self.get("/v1/models")["data"][0]["id"]
                return self
            except (urllib.error.URLError, ConnectionError):
                time.sleep(1)
        raise RuntimeError("server did not load in time")

    def __exit__(self, *_):
        self.process.terminate()
        self.process.wait(SHUTDOWN_S)
        self.log.close()

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=10) as response:
            return json.load(response)

    def complete(self, text, tokens):
        """Return (body, error) for one greedy chat completion."""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": tokens,
            "temperature": 0,
            "seed": SEED,
        }
        request = urllib.request.Request(
            self.url + "/v1/chat/completions",
            json.dumps(body).encode(),
            {"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_LIMIT_S) as response:
                return json.load(response), None
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            return None, repr(error)

    def device(self):
        return self.get("/native/v1/observability")["metrics"]["gpu_execution"]


def serve_pair_side(server, levels, reps):
    outputs = []
    for prompt in PROMPTS:
        body, error = server.complete(prompt, GREEDY_TOKENS)
        message = body["choices"][0]["message"] if body else None
        outputs.append(error or [message["content"], message.get("reasoning_content")])
    run = {"greedy_sha": hashlib.sha256(json.dumps(outputs).encode()).hexdigest(), "throughput": {}}
    for lanes in levels:
        rates, errors = [], 0
        # The first repetition warms the pool for this width and is not counted.
        for rep in range(reps + 1):
            start = time.perf_counter()
            with ThreadPoolExecutor(lanes) as pool:
                results = list(pool.map(
                    lambda i: server.complete(f"{i}-{rep}: {FILLER}", LOAD_TOKENS), range(lanes)))
            wall = time.perf_counter() - start
            tokens = sum(body["usage"]["completion_tokens"] for body, _ in results if body)
            errors += sum(1 for _, error in results if error)
            if rep:
                rates.append(tokens / wall)
        run["throughput"][lanes] = {"tok_s": statistics.median(rates), "errors": errors}
    device = server.device()
    run["step_us"] = device["sum_us"] / max(device["count"], 1)
    return run


def run_side(args, binary, cache, name, model, extra, out):
    log = out / f"{name}.log"
    with Server(binary, model, cache, log, extra) as server:
        run = serve_pair_side(server, args.levels, args.reps)
    text = log.read_text(errors="replace")
    hits, bad = tuning_hits(text)
    run.update(measured=measured(text), hits=dict(hits), bad=dict(bad))
    return run


def parse_config(text):
    """`name=model[ extra serve flags]`, e.g. `mtp=/models/27B --num-speculative-tokens 2`."""
    name, _, rest = text.partition("=")
    model, *extra = rest.split()
    if not name or not model:
        raise argparse.ArgumentTypeError(f"expected name=model [flags], got {text!r}")
    return name, model, extra


def pair(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report, failed = {}, False
    for name, model, extra in args.config:
        candidate = run_side(args, args.candidate, args.cache, f"candidate-{name}", model, extra, out)
        base = run_side(args, args.base, args.base_cache or args.cache, f"base-{name}", model, extra,
                        out)
        reasons = verdict(candidate, base)
        failed |= bool(reasons)
        report[name] = {"candidate": candidate, "base": base, "differences": reasons}
        rates = " ".join(
            f"C{lanes} {candidate['throughput'][lanes]['tok_s']:.1f}/"
            f"{base['throughput'][lanes]['tok_s']:.1f}" for lanes in args.levels)
        print(f"{name}: {candidate['greedy_sha'][:HASH_CHARS]} / {base['greedy_sha'][:HASH_CHARS]}"
              f" tok/s {rates} measured {candidate['measured']}/{base['measured']} "
              + ("OK" if not reasons else "DIFF: " + "; ".join(reasons)), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return 1 if failed else 0


def pressure_side(binary, cache, model, args, log):
    with Server(binary, model, cache, log, []) as server:
        def one(i):
            body, error = server.complete(f"{i}: {FILLER * args.prompt_repeats}", args.tokens)
            if error:
                return {"error": error}
            message = body["choices"][0]["message"]
            text = json.dumps([message["content"], message.get("reasoning_content")])
            return {"sha": hashlib.sha256(text.encode()).hexdigest(),
                    "tokens": body["usage"]["completion_tokens"]}
        start = time.perf_counter()
        with ThreadPoolExecutor(args.requests) as pool:
            results = list(pool.map(one, range(args.requests)))
        metrics = server.get("/native/v1/observability")["metrics"]
    return {"results": results, "wall_s": round(time.perf_counter() - start, 2),
            "metrics": {key: metrics.get(key) for key in PRESSURE_METRICS}}


def pressure(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sides = {name: pressure_side(binary, args.cache, args.model, args, out / f"{name}.log")
             for name, binary in (("candidate", args.candidate), ("base", args.base))}
    (out / "pressure.json").write_text(json.dumps(sides, indent=1))
    reasons = []
    for name, side in sides.items():
        errors = [result["error"] for result in side["results"] if "error" in result]
        if errors:
            reasons.append(f"{name}: {len(errors)} failed, first {errors[0]}")
    pairs = zip(sides["candidate"]["results"], sides["base"]["results"])
    differ = sum(1 for a, b in pairs if a.get("sha") != b.get("sha"))
    if differ:
        reasons.append(f"{differ} outputs differ")
    for name, side in sides.items():
        tokens = sum(result.get("tokens", 0) for result in side["results"])
        print(f"{name}: {tokens} tokens in {side['wall_s']} s {json.dumps(side['metrics'])}")
    print("OK" if not reasons else "DIFF: " + "; ".join(reasons), flush=True)
    return 1 if reasons else 0


def gap(args):
    with Server(args.binary, args.model, args.cache, Path(args.out), []) as server:
        for lanes in args.levels:
            with ThreadPoolExecutor(lanes) as pool:
                list(pool.map(lambda i: server.complete(f"{i}: {FILLER}", WARM_TOKENS),
                              range(lanes)))
            before, start = server.device(), time.perf_counter()
            with ThreadPoolExecutor(lanes) as pool:
                results = list(pool.map(lambda i: server.complete(f"{i}: {FILLER}", GAP_TOKENS),
                                        range(lanes * GAP_REQUESTS_PER_LANE)))
            wall, after = time.perf_counter() - start, server.device()
            device = (after["sum_us"] - before["sum_us"]) / 1e6
            steps = max(after["count"] - before["count"], 1)
            tokens = sum(body["usage"]["completion_tokens"] for body, _ in results if body)
            print(json.dumps({
                "c": lanes,
                "busy": round(device / wall, 3),
                "tok_s": round(tokens / wall, 1),
                "steps": steps,
                "step_ms": round(1e3 * device / steps, 2),
                "gap_per_step_ms": round(1e3 * (wall - device) / steps, 3),
            }), flush=True)
    return 0


def levels(text):
    return [int(level) for level in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    paired = commands.add_parser("pair", help="candidate against base, per configuration")
    paired.add_argument("--candidate", required=True)
    paired.add_argument("--base", required=True)
    paired.add_argument("--cache", required=True, help="warm XDG cache shared by both sides")
    paired.add_argument("--base-cache", help="separate warm cache for the base binary")
    paired.add_argument("--config", type=parse_config, action="append", required=True)
    paired.add_argument("--levels", type=levels, default=[1, 2])
    paired.add_argument("--reps", type=int, default=2)
    paired.add_argument("--out", required=True)
    paired.set_defaults(run=pair)
    loaded = commands.add_parser("pressure", help="KV oversubscription, candidate against base")
    loaded.add_argument("--candidate", required=True)
    loaded.add_argument("--base", required=True)
    loaded.add_argument("--cache", required=True)
    loaded.add_argument("--model", required=True)
    loaded.add_argument("--requests", type=int, default=64)
    loaded.add_argument("--tokens", type=int, default=1024)
    loaded.add_argument("--prompt-repeats", type=int, default=8)
    loaded.add_argument("--out", required=True)
    loaded.set_defaults(run=pressure)
    busy = commands.add_parser("gap", help="device-busy share under closed-loop load")
    busy.add_argument("--binary", required=True)
    busy.add_argument("--model", required=True)
    busy.add_argument("--cache", required=True)
    busy.add_argument("--levels", type=levels, default=[1, 4, 8, 16])
    busy.add_argument("--out", required=True, help="server log path")
    busy.set_defaults(run=gap)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
