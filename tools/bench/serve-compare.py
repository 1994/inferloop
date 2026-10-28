#!/usr/bin/env python3
"""Run one engine over a prepared serving workload and record a gate-ready report.

This promotes the local qualification experiment into a reusable harness. It
accepts the engine, the model package and the executable as arguments, feeds
every engine the identical token arrays from a workload file produced by
``serve-workloads.py``, and writes a report that ``compare-results.py`` gates.

The harness starts exactly one server per invocation, measures the case matrix
with one excluded warmup plus measured repeats, and refuses to report success
when a reference source silently substituted a weaker configuration: a server
that exits, a stream that ends early, a warm request without observed prefix
reuse, or a native measurement reported as unsuccessful all fail the run.

Examples:
    python3 tools/bench/serve-compare.py --engine native \
        --model /home/r/models/Qwen3.8-27B-NVFP4 \
        --executable target/x86_64-unknown-linux-gnu/release/infer \
        --inputs artifacts/workloads/27b-inputs.json \
        --mtp 2 --output-dir artifacts/perf
    python3 tools/bench/serve-compare.py --engine vllm \
        --model /home/r/models/Qwen3.8-27B-NVFP4 \
        --executable artifacts/vllm-compare/bin/python \
        --inputs artifacts/workloads/27b-inputs.json \
        --mtp 2 --output-dir artifacts/perf --gpu-memory-utilization 0.85
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

GROUP_START_TIMEOUT_SECONDS = 60
CASES = ["short", "long", "batch4", "hot_long"]
NATIVE_PREFIX_METRICS = ("prefix_tokens_reused",)
REFERENCE_PREFIX_METRICS = ("vllm:prefix_cache_hits_total", "sglang:prefix_cache_hits_total")
REPO_ROOT = Path(__file__).resolve().parents[2]
# Model artifacts whose content decides the input identity. Weight shards are recorded by name
# and size instead of hashed, because a 27B package is tens of gigabytes.
MODEL_ARTIFACTS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "chat_template.jinja",
)
# Section-name prefixes that only exist in a debug or explicitly unstripped image. Scanning the
# bytes for these names is not enough: std's backtrace symbolizer mentions them in read-only data
# even when the section table has none, so the section table itself is read.
DEBUG_PREFIXES = ("__debug_", ".debug_")
# Keep reference-library fan-out bounded inside the protected task limit. Existing operator
# choices remain authoritative, and only these non-secret values enter the evidence report.
REFERENCE_PROCESS_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def http(url, payload=None, timeout=600):
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def load_checklist_module():
    """Load the checklist helper, which is a hyphenated script and not importable by name."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "experiment_checklist", Path(__file__).with_name("experiment-checklist.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def eos_tokens(model):
    configuration = json.loads((Path(model) / "generation_config.json").read_text())
    value = configuration["eos_token_id"]
    return value if isinstance(value, list) else [value]


def git_identity():
    """Full source revision and whether the measured tree is dirty."""
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


# The knobs the baseline contract names, with cargo's release defaults. A report that records only
# the keys someone happened to write in the manifest cannot show the effective optimization level or
# whether debug assertions were on, which is exactly what the contract asks it to show.
RELEASE_PROFILE_DEFAULTS = {
    "opt-level": 3,
    "lto": False,
    "codegen-units": 16,
    "debug": False,
    "debug-assertions": False,
    "overflow-checks": False,
    "panic": "unwind",
    "incremental": False,
    "strip": "none",
}
# Flags that change what gets built without touching the manifest; recorded when set, because the
# contract says any override has to be visible.
PROFILE_ENVIRONMENT = ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CUDAARCHS", "NVCC_CCBIN")


def release_profile():
    """The effective release profile: the manifest's settings over cargo's defaults."""
    try:
        manifest = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {"source": "unavailable", "effective": {}, "explicit": {}}
    explicit = dict(manifest.get("profile", {}).get("release", {}))
    return {
        "source": "Cargo.toml [profile.release] over cargo's release defaults",
        "effective": {**RELEASE_PROFILE_DEFAULTS, **explicit},
        "explicit": explicit,
        "environment": {
            name: os.environ[name] for name in PROFILE_ENVIRONMENT if os.environ.get(name)
        },
    }


def elf_section_names(data):
    """Section names of an ELF image, or None when the bytes are not ELF."""
    if data[:4] != b"\x7fELF":
        return None
    is64, little = data[4] == 2, data[5] == 1
    order = "little" if little else "big"

    def read(offset, size):
        return int.from_bytes(data[offset : offset + size], order)

    if is64:
        table_offset, entry_size, count, strings = (
            read(0x28, 8),
            read(0x3A, 2),
            read(0x3C, 2),
            read(0x3E, 2),
        )
        name_at, offset_at, size_at = 0, 0x18, 0x20
        wide = 8
    else:
        table_offset, entry_size, count, strings = (
            read(0x20, 4),
            read(0x2E, 2),
            read(0x30, 2),
            read(0x32, 2),
        )
        name_at, offset_at, size_at = 0, 0x10, 0x14
        wide = 4
    if not count or strings >= count:
        return []

    def entry(index):
        base = table_offset + index * entry_size
        return read(base + name_at, 4), read(base + offset_at, wide), read(base + size_at, wide)

    _, strings_offset, strings_size = entry(strings)
    names = []
    for index in range(count):
        start, _, _ = entry(index)
        section = data[strings_offset : strings_offset + strings_size]
        end = section.find(b"\0", start)
        names.append(section[start : end if end != -1 else len(section)].decode("ascii", "replace"))
    return names


def macho_section_names(data):
    """Section names of a Mach-O image, or None when the bytes are not Mach-O."""
    magic = int.from_bytes(data[:4], "little")
    if magic not in (0xFEEDFACF, 0xFEEDFACE):
        return None
    is64 = magic == 0xFEEDFACF
    commands = int.from_bytes(data[0x10:0x14], "little")
    offset = 0x20 if is64 else 0x1C
    names = []
    for _ in range(commands):
        command = int.from_bytes(data[offset : offset + 4], "little")
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        if command == 0x19 and is64:
            count, section, stride = (
                int.from_bytes(data[offset + 64 : offset + 68], "little"),
                offset + 72,
                80,
            )
        elif command == 0x1 and not is64:
            count, section, stride = (
                int.from_bytes(data[offset + 48 : offset + 52], "little"),
                offset + 56,
                68,
            )
        else:
            count = 0
        for _ in range(count):
            names.append(data[section : section + 16].rstrip(b"\0").decode("ascii", "replace"))
            section += stride
        offset += size
    return names


def binary_identity(path):
    """Observable build facts for an engine executable.

    A debug or explicitly unstripped build carries debug sections in its section table, so the
    table is read rather than searched for names. An image whose format cannot be read is
    reported as unverifiable, and the gate refuses to call that a release build.
    """
    data = Path(path).read_bytes()
    names = elf_section_names(data)
    image = "elf"
    if names is None:
        names = macho_section_names(data)
        image = "macho"
    if names is None:
        return {
            "path": str(path),
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "image": "unknown",
            "debug_sections": [],
            "release_like": None,
        }
    debug_sections = sorted(name for name in names if name.startswith(DEBUG_PREFIXES))
    return {
        "path": str(path),
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "image": image,
        "debug_sections": debug_sections,
        "release_like": not debug_sections,
    }


def hardware_identity():
    """Device identity and driver, so a different GPU cannot be paired silently."""
    fields = ["uuid", "name", "driver_version"]
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"devices": []}
    return {
        "devices": [
            dict(zip(fields, [part.strip() for part in line.split(",")], strict=True))
            for line in raw.splitlines()
            if line.strip()
        ]
    }


def model_identity(model):
    """Fingerprint the artifacts that decide which model is really being measured."""
    files = {}
    for name in MODEL_ARTIFACTS:
        path = Path(model) / name
        if path.exists():
            files[name] = sha256_file(path)
    shards = sorted(Path(model).glob("*.safetensors"))
    return {
        "path": str(model),
        "files": files,
        "shards": {path.name: path.stat().st_size for path in shards},
    }


def serving_plan_readback(text):
    """Recover only the final server's accepted CUDA pool, never a preparation probe.

    The startup log is part of the retained benchmark evidence. A new load, replan,
    or incomplete capture invalidates earlier evidence instead of guessing a plan.
    """
    pending = accepted = None
    for raw in text.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        if "infer::load: device budget " in line:
            pending = accepted = None
        fields = {key: int(value) for key, value in re.findall(r"\b([a-z_]+)=(\d+)\b", line)}
        if "planned pooled CUDA serving storage" in line:
            accepted = None
            required = ("planned_width", "prompt_width", "quoted_bytes", "capacity", "kv_tokens")
            pending = fields if all(field in fields for field in required) else None
        elif "CUDA slot pool captured within serving envelope" in line:
            required = ("width", "capacity", "quoted_bytes", "allocated_bytes", "envelope_bytes")
            accepted = None
            if pending and all(field in fields for field in required):
                matches = (
                    fields["width"] == pending["planned_width"]
                    and fields["capacity"] == pending["capacity"]
                    and fields["quoted_bytes"] == pending["quoted_bytes"]
                    and fields["allocated_bytes"] <= fields["envelope_bytes"]
                )
                if matches:
                    accepted = {
                        "source": "native server log",
                        **{field: fields[field] for field in required},
                        "kv_tokens": pending["kv_tokens"],
                        "prompt_width": pending["prompt_width"],
                    }
        elif any(
            message in line
            for message in (
                "pooled CUDA capture exceeded budget",
                "CUDA slot pool disabled",
                "CUDA slot pool deferred",
                "retired idle CUDA request pool",
                "reclaimed idle incompatible CUDA slot pools before admission",
            )
        ):
            pending = accepted = None
    return accepted


def runtime_readback(args, base, log_path=None):
    """Read the configuration the native server actually applied."""
    if args.engine != "native":
        return {"source": "command line", "readback": None}
    try:
        with http(base + "/native/v1/runtime", timeout=30) as response:
            inspection = json.loads(response.read())
    except (OSError, ValueError, urllib.error.HTTPError) as error:
        return {"source": "unavailable", "error": repr(error)}
    readback = {
        key: inspection.get(key)
        for key in ("model_name", "lengths", "scheduler", "execution_profile", "kv_cache")
    }
    if log_path is not None:
        readback["serving_plan"] = serving_plan_readback(log_path.read_text())
    return {"source": "native runtime inspection", "readback": readback}


def native_command(args, port):
    command = [
        str(args.executable),
        str(args.model),
        "--listen",
        f"127.0.0.1:{port}",
        "--num-speculative-tokens",
        str(args.mtp),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        # The reference engine is given these; leaving native on its own defaults would compare
        # two different service constraints. The per-step token budget is deliberately not forced
        # here: vLLM treats it as a cap while native derives its own chunk and only floors it, so
        # both sides record their effective value instead of pretending one flag means one thing.
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]
    if not getattr(args, "vision", False):
        command += ["--limit-mm-per-prompt", '{"image":0,"video":0}']
    if args.extra:
        command += args.extra
    return command


def vllm_command(args, port):
    command = [
        str(args.executable),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--generation-config",
        "vllm",
    ]
    if not getattr(args, "vision", False):
        command += ["--limit-mm-per-prompt", '{"image":0,"video":0}']
    command.append("--enable-prefix-caching" if args.prefix_cache else "--no-enable-prefix-caching")
    if args.mtp:
        command += [
            "--speculative-config",
            json.dumps({"method": "mtp", "num_speculative_tokens": args.mtp}),
        ]
    if args.extra:
        command += args.extra
    return command


def sglang_command(args, port):
    command = [
        str(args.executable),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--context-length",
        str(args.max_model_len),
        "--max-running-requests",
        str(args.max_num_seqs),
        "--mem-fraction-static",
        str(args.gpu_memory_utilization),
    ]
    if args.mtp:
        command += ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", str(args.mtp)]
    if args.extra:
        command += args.extra
    return command


def read_metrics(args, base):
    route = "/native/v1/observability" if args.engine == "native" else "/metrics"
    with http(base + route) as response:
        raw = response.read().decode()
    if args.engine == "native":
        return json.loads(raw)
    result = {}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        match = re.match(r"([^ {]+)(?:\{[^}]*\})?\s+([^ ]+)", line)
        if match and ("prefix_cache" in match[1] or "spec_decode" in match[1]):
            result[match[1]] = result.get(match[1], 0.0) + float(match[2])
    return result


def prefix_reuse(args, trial):
    names = NATIVE_PREFIX_METRICS if args.engine == "native" else REFERENCE_PREFIX_METRICS
    total = 0.0
    for name in names:
        before = trial["metrics_before"]
        after = trial["metrics_after"]
        if args.engine == "native":
            left = before.get("metrics", {}).get(name, 0.0)
            right = after.get("metrics", {}).get(name, 0.0)
        else:
            left = before.get(name, 0.0)
            right = after.get(name, 0.0)
        total += right - left
    return total


def _seconds(microseconds):
    """Convert an optional microsecond reading to seconds."""
    return None if microseconds is None else microseconds / 1_000_000


def stream_request(args, base, model, eos, counter, row, *, start_barrier=None):
    if args.engine == "native":
        payload = {
            "id": next(counter),
            "model": 1,
            "input": {"Sequence": {"tokens": row["tokens"]}},
            "workload": {"Generate": {"max_new_tokens": args.tokens}},
            "sampling": {"temperature": 0.0, "eos_tokens": eos},
        }
        route = "/native/v1/stream"
    else:
        payload = {
            "model": str(model),
            "prompt": row["tokens"],
            "max_tokens": args.tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "seed": 0,
            "stop_token_ids": eos,
            "stream": True,
            "return_token_ids": True,
        }
        route = "/v1/completions"
    # Serialization and worker startup precede the common release. Record the actual client
    # send boundary separately: a barrier cannot guarantee zero OS/network scheduling skew.
    request = urllib.request.Request(
        base + route,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    if start_barrier is not None:
        start_barrier.wait()
    start_unix_ns = time.time_ns()
    started_ns = time.perf_counter_ns()
    started = started_ns / 1e9
    tokens = []
    arrivals = []
    events = []
    finished = None
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                item = json.loads(data)
                if args.engine == "native":
                    if "Token" in item:
                        new = [item["Token"]["token"]]
                    elif "Finished" in item:
                        finished = item["Finished"]
                        new = []
                    else:
                        raise RuntimeError(f"unexpected native event: {item}")
                else:
                    if "error" in item:
                        raise RuntimeError(str(item))
                    choices = item.get("choices", [])
                    new = (choices[0].get("token_ids") or []) if choices else []
                    if choices and choices[0].get("finish_reason"):
                        finished = choices[0]["finish_reason"]
                if new:
                    at = time.perf_counter() - started
                    tokens.extend(new)
                    arrivals.extend([at] * len(new))
                    events.append({"at_seconds": at, "tokens": len(new)})
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error
    ended_ns = time.perf_counter_ns()
    end_unix_ns = time.time_ns()
    elapsed = (ended_ns - started_ns) / 1e9
    if not tokens or finished is None:
        raise RuntimeError(f"incomplete stream: {tokens}, {finished}")
    if args.engine == "native" and not finished["measurement"]["successful"]:
        raise RuntimeError(f"unsuccessful native measurement: {finished}")
    visible = [(token, at) for token, at in zip(tokens, arrivals, strict=True) if token not in eos]
    # A single visible token has no inter-token interval, so TPOT is unavailable rather than
    # invalid; the gate keeps the request and reports the metric as not measurable.
    tpot = (visible[-1][1] - visible[0][1]) / (len(visible) - 1) if len(visible) > 1 else None
    finish_reason = finished.get("reason") if isinstance(finished, dict) else finished
    # The server's own view of the same request, recorded next to the client's so client latency
    # and engine phases are never conflated.
    measurement = (finished or {}).get("measurement") if isinstance(finished, dict) else None
    server = (
        {
            "ttft_seconds": _seconds(measurement.get("ttft_us")),
            "e2e_seconds": _seconds(measurement.get("e2e_us")),
            "max_tpot_seconds": _seconds(measurement.get("max_tpot_us")),
            "output_tokens": measurement.get("output_tokens"),
        }
        if measurement
        else None
    )
    return {
        "case": row["case"],
        "repeat": row["repeat"],
        "slot": row["slot"],
        "request_id": payload["id"] if args.engine == "native" else None,
        "input_tokens": len(row["tokens"]),
        "request_start_unix_ns": start_unix_ns,
        "request_start_monotonic_ns": started_ns,
        "request_end_unix_ns": end_unix_ns,
        "request_end_monotonic_ns": ended_ns,
        "token_ids": tokens,
        "output_tokens_excluding_eos": len(visible),
        # Three distinct endpoints, named separately so they cannot be mixed up: the request
        # wall, the last visible token, and (per trial) the group makespan.
        "wall_seconds": elapsed,
        "request_wall_seconds": elapsed,
        "time_to_last_token_seconds": visible[-1][1] if visible else None,
        "ttft_seconds": arrivals[0],
        "tpot_seconds": tpot,
        "tpot_available": tpot is not None,
        "arrival_seconds": arrivals,
        "stream_events": events,
        "finish_reason": finish_reason,
        "truncated": finish_reason is None,
        "server_measurement": server,
        "finished": finished,
    }


def build_command(args, port):
    if args.engine == "native":
        return native_command(args, port)
    if args.engine == "vllm":
        return vllm_command(args, port)
    return sglang_command(args, port)


def wait_ready(args, process, base, log_path, timeout):
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early; see {log_path}")
        try:
            with http(base + "/health", timeout=2) as response:
                health = response.read()
                if args.engine != "native" or json.loads(health)["ready"]:
                    return time.perf_counter() - started
        except (OSError, ValueError):
            pass
        time.sleep(0.3)
    raise RuntimeError(f"startup timeout after {timeout}s; see {log_path}")


def attribute_telemetry(report, telemetry):
    """Attach the hardware evidence for each trial's own window.

    Without this a report can show a latency difference with no way to tell an engine improvement
    from an idle device or a CPU-bound stall.
    """
    samples = telemetry["samples"]
    windows = [
        (trial["start_unix"], trial["end_unix"])
        for trial in report["trials"]
        if trial.get("start_unix") is not None and trial.get("end_unix") is not None
    ]
    for trial in report["trials"]:
        start, end = trial.get("start_unix"), trial.get("end_unix")
        if start is None or end is None:
            continue
        trial["gpu"] = gpu_summary(samples, [(start, end)])
        trial["server_process"] = process_summary(samples, [(start, end)])
    return windows


def measure_group(args, base, eos, counter, group):
    """Release serialized requests together; retain the observed send skew and endpoints."""
    released = {}

    def release():
        released["unix_ns"] = time.time_ns()
        released["monotonic_ns"] = time.perf_counter_ns()

    barrier = threading.Barrier(len(group), action=release, timeout=GROUP_START_TIMEOUT_SECONDS)

    def run(row):
        try:
            return stream_request(args, base, args.model, eos, counter, row, start_barrier=barrier)
        except BaseException:
            # A payload error must unblock peers already waiting for the common start.
            barrier.abort()
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as pool:
        futures = [pool.submit(run, row) for row in group]
        try:
            results = [future.result() for future in futures]
        except BaseException as error:
            barrier.abort()
            errors = [future.exception() for future in futures]
            # Keep the original preparation/network error rather than a peer's barrier error.
            cause = next(
                (e for e in errors if e and not isinstance(e, threading.BrokenBarrierError)),
                error,
            )
            if cause is error:
                raise
            raise cause from error
    ends = [r["request_end_monotonic_ns"] for r in results]
    starts = [r["request_start_monotonic_ns"] for r in results]
    makespan = (max(ends) - released["monotonic_ns"]) / 1e9
    return {
        "start_unix": released["unix_ns"] / 1e9,
        "end_unix": max(r["request_end_unix_ns"] for r in results) / 1e9,
        "group_start_monotonic_ns": released["monotonic_ns"],
        "start_synchronization": "barrier_after_payload_serialization",
        "request_start_skew_seconds": (max(starts) - min(starts)) / 1e9,
        "wall_seconds": makespan,
        "group_makespan_seconds": makespan,
        "results": results,
    }


def workload_cases(rows):
    """Preserve declared case order without restricting profiles to the original four cases."""
    return list(dict.fromkeys(row["case"] for row in rows))


def measure(args, report, base, rows, save):
    counter = iter(range(1, 1_000_000))
    eos = report["eos_tokens"]
    for case in workload_cases(rows):
        declared = [r for r in rows if r["case"] == case]
        if not declared:
            continue
        for repeat in range(-1, args.repeats):
            group = [r for r in declared if r["repeat"] == repeat]
            # The workload declares how many requests a case runs together; measuring fewer
            # would silently turn batch4 into four single-request runs.
            expected = declared[0]["concurrency"]
            if len(group) != expected or len({r["slot"] for r in group}) != len(group):
                raise RuntimeError(
                    f"{case} repeat {repeat} has {len(group)} requests, expected {expected}"
                )
            before = read_metrics(args, base)
            timing = measure_group(args, base, eos, counter, group)
            results = timing["results"]
            makespan = timing["group_makespan_seconds"]
            trial = {
                "case": case,
                "repeat": repeat,
                "warmup": repeat < 0,
                "concurrency": len(group),
                "declared_concurrency": expected,
                **timing,
            }
            time.sleep(1.1)
            trial["metrics_before"] = before
            trial["metrics_after"] = read_metrics(args, base)
            # Reuse is recorded per trial, so one hit cannot stand in for the whole hot case.
            trial["prefix_tokens_reused"] = prefix_reuse(args, trial)
            report["trials"].append(trial)
            save()
            outputs = [r["output_tokens_excluding_eos"] for r in results]
            reused = trial["prefix_tokens_reused"]
            print(f"{case} {repeat} {makespan:.3f}s reuse={reused:.0f} {outputs}", flush=True)


def load_workload(path, repeats, max_input_tokens=0):
    """Read and validate a workload file, returning its rows.

    A partially filtered file must not silently measure an empty concurrency
    group, so every case that appears must carry its excluded warmup plus all
    measured repeats.
    """
    rows = json.loads(Path(path).read_text())
    for row in rows:
        tokens = row.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise SystemExit(f"workload row has no tokens: {row.get('case')}/{row.get('slot')}")
        if not isinstance(row.get("concurrency"), int) or row["concurrency"] < 1:
            raise SystemExit(
                "workload row declares no positive concurrency: "
                f"{row.get('case')}/{row.get('slot')}"
            )
        if max_input_tokens and len(tokens) > max_input_tokens:
            raise SystemExit(
                f"input of {len(tokens)} tokens exceeds --max-input-tokens {max_input_tokens}"
            )
    for case in workload_cases(rows):
        declared = {r["concurrency"] for r in rows if r["case"] == case}
        if not declared:
            continue
        if len(declared) > 1:
            raise SystemExit(f"{case} declares inconsistent concurrency: {sorted(declared)}")
        expected = declared.pop()
        for repeat in range(-1, repeats):
            count = sum(1 for r in rows if r["case"] == case and r["repeat"] == repeat)
            if count != expected:
                raise SystemExit(
                    f"{case} repeat {repeat} carries {count} slots, expected {expected}"
                )
    missing = [
        f"{case}:{repeat}"
        for case in workload_cases(rows)
        for repeat in range(-1, repeats)
        if not any(r["case"] == case and r["repeat"] == repeat for r in rows)
    ]
    if missing:
        raise SystemExit(f"workload is missing case/repeat rows: {', '.join(missing)}")
    return rows


def workload_matrix(rows, repeats):
    """The case/slot/repeat matrix the workload declares, so the gate can require all of it."""
    return {
        case: {
            "concurrency": next(r["concurrency"] for r in rows if r["case"] == case),
            "repeats": repeats,
        }
        for case in workload_cases(rows)
    }


def require_protected_run():
    """Refuse to measure outside the repository's memory-limited benchmark scope.

    The plan and the telemetry tool both require it: an unprotected job can push the host into
    swap, the run is then not comparable to a protected baseline, and the review found the
    previous numbers had been taken outside it.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_limits", Path(__file__).with_name("check_limits.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.limits()
    except (RuntimeError, OSError, StopIteration) as error:
        raise RuntimeError(
            f"{error}. Measure with: bash tools/bench/safe-run.sh python3 "
            "tools/bench/serve-compare.py ..."
        ) from error


TELEMETRY_INTERVAL_SECONDS = 0.1
TELEMETRY_FIELDS = (
    "timestamp",
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.current.memory",
)


def start_telemetry(output, pid, interval=TELEMETRY_INTERVAL_SECONDS):
    """Attach the repository telemetry tool to the server for the measured window.

    A measurement without hardware evidence cannot show whether a difference came from the engine
    or from the device, so this is part of the run rather than an optional extra.
    """
    monitor = Path(__file__).with_name("hardware-monitor.py")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                str(monitor),
                "--pid",
                str(pid),
                "--output",
                str(output),
                "--interval",
                str(interval),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:  # pragma: no cover - depends on the host
        return None, str(error)
    return process, None


def stop_telemetry(process):
    """Stop the monitor and return anything it complained about."""
    if process is None:
        return None
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        process.kill()
        process.wait(timeout=10)
    if process.returncode in (0, None) or process.stderr is None:
        return None
    try:
        return process.stderr.read().strip() or None
    except (OSError, ValueError):  # pragma: no cover - defensive
        return None


def load_telemetry(path):
    """Parse the telemetry log into timestamped GPU rows and process samples."""
    metadata, samples = {}, []
    text = Path(path).read_text() if Path(path).is_file() else ""
    for line in text.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("type") == "metadata":
            metadata = entry
        elif entry.get("type") == "sample":
            fields = metadata.get("gpu_fields") or list(TELEMETRY_FIELDS)
            rows = []
            for row in entry.get("gpu") or []:
                if len(row) != len(fields):
                    continue
                record = {}
                for name, value in zip(fields, row, strict=True):
                    if name in ("timestamp", "pstate", "clocks_event_reasons.active"):
                        continue
                    try:
                        record[name] = float(value)
                    except ValueError:
                        record = None
                        break
                if record is not None:
                    rows.append(record)
            samples.append(
                {
                    "unix_seconds": entry.get("unix_seconds"),
                    "gpu": rows,
                    "cpu_percent_one_core": entry.get("cpu_percent_one_core") or {},
                }
            )
    return {"metadata": metadata, "samples": samples}


def within(sample, windows):
    """Whether a sample falls inside any of the measurement windows."""
    return sample["unix_seconds"] is not None and any(
        start <= sample["unix_seconds"] <= end for start, end in windows
    )


def gpu_summary(samples, windows):
    """Telemetry for the busiest device over the union of the given windows.

    Passing only trial windows measures the device while it is working; passing the whole measured
    span would dilute the mean with the idle gaps between groups. The busiest device is the one
    whose utilization best explains the window, so a low value is real headroom on the GPU that
    did the work rather than an idle sibling.
    """
    window = [sample for sample in samples if within(sample, windows)]
    rows = [row for sample in window for row in sample["gpu"]]
    if not rows:
        return None
    by_device = {}
    for row in rows:
        by_device.setdefault(int(row.get("index", 0)), []).append(row)
    index, device = max(
        by_device.items(),
        key=lambda item: statistics.fmean(row.get("utilization.gpu", 0.0) for row in item[1]),
    )

    def mean(field):
        return statistics.fmean(row.get(field, 0.0) for row in device)

    def peak(field):
        return max(row.get(field, 0.0) for row in device)

    return {
        "device": index,
        "samples": len(device),
        "utilization_gpu_mean": mean("utilization.gpu"),
        "utilization_gpu_max": peak("utilization.gpu"),
        "utilization_memory_mean": mean("utilization.memory"),
        "memory_used_max_mib": peak("memory.used"),
        "power_mean_w": mean("power.draw"),
        "sm_clock_mean_mhz": mean("clocks.current.sm"),
        "temperature_max_c": peak("temperature.gpu"),
    }


def process_summary(samples, windows):
    """Server process CPU over the union of the given windows.

    A busy CPU alongside an under-used GPU is what a host-bound step looks like, so it is measured
    over the same windows as the device.
    """
    window = [
        sample for sample in samples if within(sample, windows) and sample["cpu_percent_one_core"]
    ]
    if not window:
        return None
    totals = [sum(sample["cpu_percent_one_core"].values()) for sample in window]
    return {
        "samples": len(window),
        "cpu_percent_one_core_mean": statistics.fmean(totals),
        "cpu_percent_one_core_max": max(totals),
    }


def require_cache_switch(args):
    """Refuse a requested cache state the engine cannot actually apply."""
    if args.engine == "native" and not args.prefix_cache:
        raise RuntimeError(
            "the native engine cannot disable its prefix cache, so --no-prefix-cache cannot take "
            "effect; a cold profile is unavailable for this engine version"
        )


def validate_workload_resources(args, rows):
    """Reject an impossible profile before an engine allocates its model or KV cache."""
    if args.tokens < 1 or args.max_model_len < 1 or args.max_num_seqs < 1:
        raise RuntimeError("output, context and sequence limits must all be positive")
    for row in rows:
        required = len(row["tokens"]) + args.tokens
        if required > args.max_model_len:
            raise RuntimeError(
                f"{row['case']}/{row['repeat']}/{row['slot']} needs {required} context tokens, "
                f"exceeding --max-model-len {args.max_model_len}"
            )
        if row["concurrency"] > args.max_num_seqs:
            raise RuntimeError(
                f"{row['case']} concurrency {row['concurrency']} exceeds "
                f"--max-num-seqs {args.max_num_seqs}"
            )


def process_environment(engine):
    """Bound reference thread fan-out and retain the exact non-secret process settings."""
    environment = os.environ.copy()
    if engine != "native":
        for name, value in REFERENCE_PROCESS_ENV.items():
            environment.setdefault(name, value)
    return environment


def run(args):
    require_cache_switch(args)
    require_protected_run()
    rows = load_workload(args.inputs, args.repeats, args.max_input_tokens)
    validate_workload_resources(args, rows)
    eos = eos_tokens(args.model)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    command = build_command(args, port)
    environment = process_environment(args.engine)
    name = f"{args.model.name}-mtp{args.mtp}-{args.engine}"
    run_id = args.run_id or None
    stem = args.output_dir / (name if run_id is None else f"{name}-{run_id}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"{stem}.server.log")
    report_path = Path(f"{stem}.json")
    # Evidence must not be replaced by a later run of the same name.
    if report_path.exists():
        raise RuntimeError(
            f"report {report_path} already exists; pass a unique --run-id or a fresh --output-dir"
        )
    build = (
        binary_identity(args.executable)
        if args.engine == "native"
        else {
            "path": str(args.executable),
            "sha256": sha256_file(args.executable),
            "release_like": None,
        }
    )
    profile = release_profile()
    identity = {
        "run_id": run_id,
        "profile_id": args.profile_id,
        "engine": args.engine,
        "engine_version": args.engine_version,
        "source": git_identity(),
        "release_profile": profile,
        "build": build,
        "model": model_identity(args.model),
        "hand_reported_package_sha256": args.package_sha256,
        "hardware": hardware_identity(),
        "limits": (
            {
                "source": "declared on the command line, verified after startup",
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
            }
            if args.engine != "native"
            else {
                "source": "pending native runtime inspection",
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
            }
        ),
        "cache": {
            "requested": bool(args.prefix_cache),
            "effective": bool(args.prefix_cache),
        },
    }
    # A release identity is verifiable for native through the binary's own sections; a debug or
    # otherwise unverifiable build must not be recorded as if it were a release.
    identity["verified"] = args.engine != "native" or build["release_like"] is True
    report = {
        "engine": args.engine,
        "engine_version": args.engine_version,
        "model": str(args.model),
        "model_package_sha256": args.package_sha256,
        "command": command,
        "max_new_tokens": args.tokens,
        "temperature": 0,
        "eos_tokens": eos,
        "mtp_depth": args.mtp,
        "prefix_cache_enabled": bool(args.prefix_cache),
        "vision_enabled": getattr(args, "vision", False),
        "binary_sha256": build.get("sha256"),
        "workload_file": str(args.inputs),
        "inputs_sha256": sha256_bytes(json.dumps(rows, sort_keys=True).encode()),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF"),
        "process_environment": {
            name: environment[name] for name in REFERENCE_PROCESS_ENV if name in environment
        },
        "matrix": workload_matrix(rows, args.repeats),
        "identity": identity,
        "trials": [],
        "completed": False,
    }

    def save():
        report_path.write_text(json.dumps(report, indent=2))

    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            report["startup_seconds"] = wait_ready(
                args, process, base, log_path, args.startup_timeout
            )
            print(f"ready after {report['startup_seconds']:.1f}s", flush=True)
            report["config_readback"] = runtime_readback(args, base, log_path)
            if args.engine == "native":
                readback = (report["config_readback"] or {}).get("readback") or {}
                lengths = readback.get("lengths") or {}
                # The CLI values are not the effective configuration; the readback is.
                report["identity"]["limits"] = {
                    "source": "native runtime inspection",
                    "effective_max_model_len": lengths.get("total"),
                    "effective_input_cap": lengths.get("input_cap"),
                    "effective_output_cap": lengths.get("output_cap"),
                    "effective_prefill_width": (readback.get("execution_profile") or {}).get(
                        "prefill_width"
                    ),
                    "declared": report["identity"]["limits"],
                }
                if lengths.get("total") != args.max_model_len:
                    raise RuntimeError(
                        "native effective context "
                        f"{lengths.get('total')} differs from the requested {args.max_model_len}"
                    )
            if args.checklist is not None:
                # Declared conditions are checked against what the run actually did, including
                # the effective limits just read back, not against the command line.
                module = load_checklist_module()
                checklist = module.load(args.checklist)
                mismatches = module.check_run(checklist, report)
                report["checklist"] = {
                    "path": str(args.checklist),
                    "sha256": sha256_file(args.checklist),
                    "profile_id": checklist["profile_id"],
                    "compliant": not mismatches,
                    "mismatches": mismatches,
                }
                save()
                if mismatches:
                    raise RuntimeError(
                        "run does not match the experiment checklist: " + "; ".join(mismatches)
                    )
            telemetry_path = Path(f"{stem}.telemetry.jsonl")
            monitor, monitor_error = start_telemetry(telemetry_path, process.pid)
            try:
                measure(args, report, base, rows, save)
            finally:
                monitor_error = stop_telemetry(monitor) or monitor_error
            telemetry = load_telemetry(telemetry_path)
            windows = attribute_telemetry(report, telemetry)
            measured_from = min((t["start_unix"] for t in report["trials"]), default=None)
            measured_to = max((t["end_unix"] for t in report["trials"]), default=None)
            span = [(measured_from, measured_to)] if measured_from is not None else []
            report["telemetry"] = {
                "log": telemetry_path.name,
                "sha256": (sha256_file(telemetry_path) if telemetry_path.is_file() else None),
                "interval_seconds": TELEMETRY_INTERVAL_SECONDS,
                "source": "tools/bench/hardware-monitor.py",
                "error": monitor_error,
                "samples": len(telemetry["samples"]),
                # While the workload runs: the answer to "is the device busy during execution".
                # The span also covers the idle gaps between groups.
                "active_window": gpu_summary(telemetry["samples"], windows) if windows else None,
                "active_process": (
                    process_summary(telemetry["samples"], windows) if windows else None
                ),
                "measured_span": gpu_summary(telemetry["samples"], span) if span else None,
            }
            save()
            hot = [t for t in report["trials"] if t["case"] == "hot_long" and not t["warmup"]]
            reused = sum(t["prefix_tokens_reused"] for t in hot)
            report["hot_prefix_tokens_reused"] = reused
            if hot and reused <= 0:
                raise RuntimeError("prefix cache enabled but no actual hot-prefix reuse observed")
            # Every measured repeat must show reuse; one hit cannot speak for the whole case.
            cold = [t["repeat"] for t in hot if t["prefix_tokens_reused"] <= 0]
            if cold:
                raise RuntimeError(
                    f"hot_long repeats {cold} reused no prefix even though the cache is enabled"
                )
            report["completed"] = True
        except Exception as error:
            report["error"] = repr(error)
            raise
        finally:
            save()
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    print(f"report {stem}.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=["native", "vllm", "sglang"])
    parser.add_argument("--model", required=True, type=Path, help="model package directory")
    parser.add_argument(
        "--executable", required=True, type=Path, help="engine executable or interpreter"
    )
    parser.add_argument(
        "--inputs", required=True, type=Path, help="workload JSON from serve-workloads.py"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=64, help="generated tokens per request")
    parser.add_argument("--mtp", type=int, default=0, help="speculative/MTP depth")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-input-tokens", type=int, default=0, help="refuse longer inputs when non-zero"
    )
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--engine-version", default=None)
    parser.add_argument("--package-sha256", default=None)
    parser.add_argument(
        "--run-id", default=None, help="unique evidence id; reports never overwrite"
    )
    parser.add_argument("--profile-id", default=None, help="experiment profile this run belongs to")
    parser.add_argument(
        "--checklist",
        type=Path,
        default=None,
        help="experiment profile this run must match; a mismatch fails the run",
    )
    parser.add_argument("--prefix-cache", action="store_true", default=True)
    parser.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false")
    parser.add_argument(
        "--vision",
        action="store_true",
        help="keep the vision tower available instead of the text-only diagnostic default",
    )
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra engine flags")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("the serving gate requires at least three measured repeats")
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be within (0, 1)")
    try:
        run(args)
    except Exception as error:  # top-level reporting boundary
        print(f"benchmark failed: {error!r}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
