#!/usr/bin/env python3
"""Run paired A/B rounds and preserve raw results before judging the native implementation.

Run under tools/bench/safe-run.sh. Build binaries and generate fixtures first;
neither compilation nor upload is included in the CUDA-event measurements.
"""

import argparse
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

from attention_gate import ROUNDS, evaluate, require


def capture(command):
    return subprocess.check_output(command, text=True).strip()


def environment():
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    require(bool(selected), "no visible CUDA device")
    gpu = capture(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader",
            f"--id={selected}",
        ]
    )
    device_uuid, name, driver = [field.strip() for field in gpu.split(",")]
    return {
        "device": f"{device_uuid} {name}",
        "device_uuid": device_uuid,
        "driver": driver,
        "toolkit": capture(["nvcc", "--version"]),
    }


def measurements(command, path, env):
    process = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    path.write_text(process.stdout + process.stderr)
    require(process.returncode == 0, f"candidate/baseline execution failed; see {path}")
    records, samples = [], None
    for line in process.stdout.splitlines():
        if "ATTENTION_BENCH " in line:
            require(samples is None, "unmatched baseline timing record")
            samples = json.loads(line.split("ATTENTION_BENCH ", 1)[1])["samples_ms"]
        if "ATTENTION_GATE " in line:
            row = json.loads(line.split("ATTENTION_GATE ", 1)[1])
            if "samples_ms" not in row:
                require(samples is not None, "missing baseline timing record")
                row["samples_ms"] = samples
            if not row["shape"]["rope"] and "core_samples_ms" not in row:
                row["core_samples_ms"] = row["samples_ms"]
            samples = None
            records.append(row)
    require(samples is None and records, "missing/unmatched gate records")
    return records


def run(args):
    fixture = args.fixtures.resolve()
    manifest = json.loads(fixture.with_suffix(".json").read_text())
    require(
        hashlib.sha256(fixture.read_bytes()).hexdigest() == manifest["fixture_sha256"],
        "fixture digest mismatch",
    )
    metadata = environment()
    run_id = str(uuid.uuid4())
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    runs = {}
    for name in ("native", "candle-f16", "candle-bf16"):
        runs[name] = {
            **metadata,
            "schema": 1,
            "build": "release",
            "implementation": name,
            "precision": "compensated-f32" if name == "native" else name.removeprefix("candle-"),
            "run_id": run_id,
            "contract": manifest["contract"],
            "scope": manifest["scope"],
            "fixture_sha256": manifest["fixture_sha256"],
            "records": [],
        }
    binaries = {"native": args.native.resolve(), "candle": args.candle.resolve()}
    for name, path in binaries.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for role in ["native"] if name == "native" else ["candle-f16", "candle-bf16"]:
            runs[role]["binary_sha256"] = digest
    env = {
        **os.environ,
        "INFER_ATTENTION_GATE": str(fixture),
        "CUDA_VISIBLE_DEVICES": metadata["device_uuid"],
    }
    commands = {
        "native": [
            str(binaries["native"]),
            "attention::gate_tests::attention_native_gate",
            "--exact",
            "--ignored",
            "--test-threads=1",
            "--nocapture",
        ],
        "candle-f16": [str(binaries["candle"]), str(fixture), "f16"],
        "candle-bf16": [str(binaries["candle"]), str(fixture), "bf16"],
    }
    for round_id in range(ROUNDS):
        # Reverse order on odd rounds to reduce systematic thermal/order bias.
        order = list(commands) if round_id % 2 == 0 else list(reversed(commands))
        for role in order:
            print(f"round {round_id + 1}/{ROUNDS}: {role}", flush=True)
            require(environment() == metadata, "device/driver changed during measurement")
            records = measurements(commands[role], args.out / f"{role}-{round_id}.log", env)
            runs[role]["records"].extend({**r, "round": round_id} for r in records)
            (args.out / f"{role}.json").write_text(
                json.dumps(runs[role], indent=2, allow_nan=False) + "\n"
            )
    for role, path in (("native", binaries["native"]), ("candle-f16", binaries["candle"])):
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == runs[role]["binary_sha256"],
            "binary changed during measurement",
        )
    require(
        hashlib.sha256(fixture.read_bytes()).hexdigest() == manifest["fixture_sha256"],
        "fixtures changed during measurement",
    )
    reports = {
        role: evaluate(manifest, runs[role], runs["native"])
        for role in ("candle-f16", "candle-bf16")
    }
    for role, report in reports.items():
        (args.out / f"{role}-decision.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            f"{role}: {report['decision']}, geomean speedup {report['geomean_speedup']:.3f}x",
            flush=True,
        )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixtures", "native", "candle", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    require(
        not any(args.out.glob("*.json")), "use a fresh output directory; existing evidence retained"
    )
    try:
        reports = run(args)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        (args.out / "invalid.json").write_text(
            json.dumps({"decision": "invalid", "error": str(error)}) + "\n"
        )
        raise SystemExit(str(error)) from error
    raise SystemExit(0 if all(r["decision"] == "pass" for r in reports.values()) else 1)


if __name__ == "__main__":
    main()
