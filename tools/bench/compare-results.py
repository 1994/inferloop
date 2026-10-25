#!/usr/bin/env python3
"""Gate paired serving benchmark JSON reports; lower latency is better.

This judges three things separately and never lets one stand in for another:

* alignment - both reports describe the same model artifact, resource limits, cache state and
  hardware, and both were produced by a verifiable release build;
* performance - the paired latency ratios against the declared threshold;
* numeric - whether the two sides produced the same tokens, when the caller requires it.

Task quality is not measured here; a report that passes says nothing about answer quality.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

# Telemetry fields a measured trial must carry; without them a latency difference cannot be read
# as engine behaviour rather than an idle device or a CPU-bound stall.
REQUIRED_TELEMETRY = ("utilization_gpu_mean", "utilization_gpu_max", "sm_clock_mean_mhz")

# Per-request latency that must be present and positive; TPOT is optional because a request with
# one visible token has no inter-token interval.
REQUIRED_REQUEST_LATENCY = ("ttft_seconds",)


def release_profile_state(identity):
    """Whether a native report's release profile is proven, legacy, or outright wrong.

    Reports written before the effective-profile rule carry only the manifest's explicit keys, so
    their profile cannot be proven - they are `legacy`, not invalid. The plan keeps old profiles'
    results for handover, so a caller may still use one as the fixed side of a comparison; a new
    measurement must never be legacy.
    """
    profile = identity.get("release_profile")
    if not isinstance(profile, dict) or not profile.get("effective"):
        return "legacy"
    effective = profile["effective"]
    if effective.get("debug-assertions") is not False:
        raise ValueError("native report was built with debug assertions")
    if effective.get("overflow-checks") is not False:
        raise ValueError("native report was built with overflow checks")
    if effective.get("debug") not in (False, 0, "0", None):
        raise ValueError("native report was built with debug information")
    level = effective.get("opt-level")
    if level in (0, "0"):
        raise ValueError("native report was built without optimization")
    if effective.get("incremental"):
        raise ValueError("native report was built incrementally")
    return "proven"


def require_release_identity(report):
    """Reject a report whose build, model or hardware identity cannot be verified."""
    identity = report.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("report has no identity block")
    if identity.get("verified") is not True:
        raise ValueError("report identity is not verified")
    engine = identity.get("engine")
    if engine == "native":
        build = identity.get("build")
        if not isinstance(build, dict) or build.get("release_like") is not True:
            raise ValueError("native report was not built as a release binary")
        if not (identity.get("source") or {}).get("revision"):
            raise ValueError("native report records no source revision")
    elif not identity.get("engine_version"):
        raise ValueError("reference report records no engine version")
    if not (identity.get("hardware") or {}).get("devices"):
        raise ValueError("report records no hardware identity")
    if not isinstance(identity.get("model"), dict) or not identity["model"].get("files"):
        raise ValueError("report records no model artifact fingerprint")


def native_resource_signature(report):
    """Read fixed startup geometry and capacity, excluding activity counters and numerics.

    CUDA's pool_bytes is the slot pool expressed in the runtime's block vocabulary,
    not measured physical GPU bytes. It still distinguishes capture replans that
    leave identical command-line limits but different admitted concurrency.
    """
    block = report.get("config_readback")
    if not isinstance(block, dict):
        raise ValueError("native resource comparison needs runtime readback")
    readback = block.get("readback")
    if not isinstance(readback, dict):
        raise ValueError("native resource comparison needs runtime readback")
    scheduler = readback.get("scheduler")
    if not isinstance(scheduler, dict) or not scheduler:
        raise ValueError("native resource comparison needs scheduler readback")
    signature = {"scheduler": scheduler}
    for section, fields in (
        (
            "execution_profile",
            (
                "prefill_width",
                "pooled_prefill_width",
                "batch_width",
                "draft_depth",
                "arena_budget_bytes",
            ),
        ),
        ("kv_cache", ("block_size", "total_blocks", "pool_bytes", "bytes_per_block")),
        (
            "serving_plan",
            ("width", "capacity", "kv_tokens", "prompt_width", "quoted_bytes", "envelope_bytes"),
        ),
    ):
        values = readback.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"native resource comparison needs {section} readback")
        for field in fields:
            value = values.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"native resource comparison needs valid {section}.{field}")
            signature[f"{section}.{field}"] = value
    return signature


def align_native_resources(baseline, candidate):
    """Require a declared fixed-resource native A/B to retain its effective plan."""
    if any(report["identity"]["engine"] != "native" for report in (baseline, candidate)):
        raise ValueError("identical native resources can only be required for native A/B")
    left, right = native_resource_signature(baseline), native_resource_signature(candidate)
    for field, value in left.items():
        if value != right[field]:
            raise ValueError(f"unaligned native resource: {field} ({value!r} != {right[field]!r})")


def align(baseline, candidate):
    """Require the two reports to describe the same experiment conditions."""
    for field in ("model", "hardware"):
        if baseline["identity"].get(field) != candidate["identity"].get(field):
            raise ValueError(f"unaligned {field} identity")
    left_cache = (baseline["identity"].get("cache") or {}).get("effective")
    right_cache = (candidate["identity"].get("cache") or {}).get("effective")
    if left_cache != right_cache:
        raise ValueError("unaligned prefix cache state")
    if baseline.get("vision_enabled") != candidate.get("vision_enabled"):
        raise ValueError("unaligned vision availability")
    if baseline.get("gpu_memory_utilization") != candidate.get("gpu_memory_utilization"):
        raise ValueError("unaligned resource constraint: gpu_memory_utilization")
    # Both sides must have matched the same declared profile, or the comparison is between two
    # different experiments that happen to share a workload file.
    left_checklist = baseline.get("checklist")
    right_checklist = candidate.get("checklist")
    if not isinstance(left_checklist, dict) or not isinstance(right_checklist, dict):
        raise ValueError("both reports must be checked against an experiment checklist")
    for side, block in (("baseline", left_checklist), ("candidate", right_checklist)):
        if block.get("compliant") is not True:
            raise ValueError(f"{side} run did not match its experiment checklist")
    if left_checklist.get("sha256") != right_checklist.get("sha256"):
        raise ValueError("the two reports were measured against different checklists")
    left_engine = baseline["identity"].get("engine")
    right_engine = candidate["identity"].get("engine")
    if left_engine == right_engine:
        # An A/B of one engine must hold the serving limits fixed.
        if baseline["identity"].get("limits") != candidate["identity"].get("limits"):
            raise ValueError("unaligned serving limits")
        return
    # Across engines the internal limits differ by construction, so the shared context ceiling is
    # checked explicitly and the experiment profile has to name the pairing.
    profile = baseline["identity"].get("profile_id")
    if not profile or profile != candidate["identity"].get("profile_id"):
        raise ValueError("cross-engine comparison needs one shared profile id")
    native, reference = (baseline, candidate) if left_engine == "native" else (candidate, baseline)
    total = (native["identity"].get("limits") or {}).get("effective_max_model_len")
    declared = (reference["identity"].get("limits") or {}).get("max_model_len")
    if total is None or declared is None or total != declared:
        raise ValueError(
            f"context limits are not aligned across engines: native {total}, reference {declared}"
        )


def orientation(baseline, candidate, reference="vllm"):
    """Order a pair so the ratio is always "engine under test / reference engine".

    The plan's target is `native/vLLM <= 1.10x`: the candidate must not be more than 10% slower
    than the reference. Which side is which therefore has to be explicit; getting it backwards
    turns a pass into a failure and hides real regressions behind a faster reference.
    """
    left = (baseline.get("identity") or {}).get("engine")
    right = (candidate.get("identity") or {}).get("engine")
    if left == right:
        # An A/B of one engine: the candidate is the later measurement.
        return candidate, baseline, f"{left}/baseline"
    if reference not in (left, right):
        raise ValueError(
            f"neither report is the reference engine {reference!r} (have {left!r}, {right!r}); "
            "name the reference side instead of assuming its direction"
        )
    if left == reference:
        return candidate, baseline, f"{right}/{reference}"
    return baseline, candidate, f"{left}/{reference}"


def measured(report):
    """Validate one report and return its measured trials keyed by case and repeat."""
    if report.get("completed") is not True:
        raise ValueError("benchmark did not complete")
    matrix = report.get("matrix")
    if not isinstance(matrix, dict) or not matrix:
        raise ValueError("report does not declare the measured matrix")
    rows = {}
    for trial in report["trials"]:
        if trial["warmup"]:
            continue
        key = (trial["case"], trial["repeat"])
        if key in rows:
            raise ValueError("duplicate trial")
        declared = matrix.get(trial["case"])
        if not isinstance(declared, dict) or "concurrency" not in declared:
            raise ValueError(f"case is not declared in the matrix: {trial['case']}")
        expected = declared["concurrency"]
        # A partially measured concurrency group is not the case it claims to be.
        if trial["concurrency"] != expected or len(trial["results"]) != expected:
            raise ValueError(
                f"{trial['case']} measured {len(trial['results'])} requests, expected {expected}"
            )
        makespan = trial.get("wall_seconds")
        if makespan is None or not math.isfinite(makespan) or makespan <= 0:
            raise ValueError("missing or invalid group makespan")
        for result in trial["results"]:
            if not result.get("finish_reason") or result.get("truncated"):
                raise ValueError("failed or truncated request")
            for field in REQUIRED_REQUEST_LATENCY:
                value = result.get(field)
                if value is None or not math.isfinite(value) or value <= 0:
                    raise ValueError("missing or invalid latency")
            # A single visible token has no inter-token interval; that is unavailable, not bad.
            tpot = result.get("tpot_seconds")
            if tpot is not None and (not math.isfinite(tpot) or tpot < 0):
                raise ValueError("invalid TPOT")
            # When the engine reports its own view, it must agree with what was streamed: a
            # server that generated more than it delivered measured a truncated request.
            server = result.get("server_measurement")
            reported = None if server is None else server.get("output_tokens")
            if reported is not None and reported != len(result["token_ids"]):
                raise ValueError(
                    f"server reported {reported} generated tokens but "
                    f"{len(result['token_ids'])} were streamed"
                )
        # Hardware evidence is part of the measurement, not an optional extra.
        telemetry = trial.get("gpu")
        if not isinstance(telemetry, dict) or any(
            telemetry.get(field) is None for field in REQUIRED_TELEMETRY
        ):
            raise ValueError(f"measured trial has no complete hardware telemetry: {trial['case']}")
        rows[key] = trial
    if not rows:
        raise ValueError("no measured trials")
    for case, declared in matrix.items():
        for repeat in range(declared.get("repeats", 0)):
            if (case, repeat) not in rows:
                raise ValueError(f"matrix row was not measured: {case}:{repeat}")
    hot = {key: trial for key, trial in rows.items() if key[0] == "hot_long"}
    if hot:
        if report.get("hot_prefix_tokens_reused", 0) <= 0:
            raise ValueError("hot cache workload has no observed prefix reuse")
        # Every measured repeat must reuse; a single hit cannot stand for the whole case.
        cold = sorted(
            key[1] for key, trial in hot.items() if trial.get("prefix_tokens_reused", 0) <= 0
        )
        if cold:
            raise ValueError(f"hot cache repeats without observed reuse: {cold}")
    return rows


def availability(values):
    """Report a metric's usable samples, and whether it can be judged at all."""
    usable = [value for value in values if value is not None]
    return usable


def compare(
    baseline,
    candidate,
    max_ratio=1.1,
    identical=False,
    reference="vllm",
    allow_legacy_baseline=False,
    require_identical_resources=False,
):
    for field in ("max_new_tokens", "temperature", "mtp_depth", "eos_tokens"):
        if field not in baseline or baseline[field] != candidate.get(field):
            raise ValueError(f"unaligned or missing workload field: {field}")
    for field in ("model", "inputs_sha256", "prefix_cache_enabled"):
        if field not in baseline or baseline[field] != candidate.get(field):
            raise ValueError(f"unaligned or missing workload field: {field}")
    require_release_identity(baseline)
    require_release_identity(candidate)
    # A new measurement has to prove its profile. The flag marks a historical comparison - a frozen
    # pair re-checked, or a new candidate against a frozen reference - where either side may predate
    # the rule, because the plan keeps old profiles' results rather than discarding them.
    legacy = [
        side
        for side in ("baseline", "candidate")
        if release_profile_state(locals()[side]["identity"]) != "proven"
    ]
    if legacy and not allow_legacy_baseline:
        raise ValueError(
            f"{' and '.join(legacy)} report predates the effective-profile rule; pass"
            " --allow-legacy-baseline to treat this as a historical comparison"
        )
    align(baseline, candidate)
    if require_identical_resources:
        align_native_resources(baseline, candidate)
    tested_report, fixed_report, label = orientation(baseline, candidate, reference)
    old, new = measured(fixed_report), measured(tested_report)
    if old.keys() != new.keys():
        raise ValueError("trial sets differ")
    token_mismatches = 0
    for key, trial in old.items():
        other = new[key]
        left = {r["slot"]: r for r in trial["results"]}
        right = {r["slot"]: r for r in other["results"]}
        if (
            len(left) != len(trial["results"])
            or len(right) != len(other["results"])
            or left.keys() != right.keys()
        ):
            raise ValueError(f"request slots differ: {key}")
        for slot, row in left.items():
            peer = right[slot]
            for item in (row, peer):
                visible = sum(token not in baseline["eos_tokens"] for token in item["token_ids"])
                if visible != item["output_tokens_excluding_eos"]:
                    raise ValueError("reported output count differs from token IDs")
            for field in ("input_tokens", "output_tokens_excluding_eos"):
                if row[field] != peer[field]:
                    raise ValueError(f"request work differs: {key}, {slot}, {field}")
            token_mismatches += row["token_ids"] != peer["token_ids"]
    # `old` is the reference side and `new` the engine under test, so every ratio below is
    # engine-under-test over reference and the threshold reads directly.
    metrics = []
    for case in sorted({key[0] for key in old}):
        groups = [[v for k, v in rows.items() if k[0] == case] for rows in (old, new)]
        if len(groups[0]) < 3:
            raise ValueError(f"at least three measured repeats required: {case}")
        for field in ("wall_seconds", "ttft_seconds", "tpot_seconds"):
            samples = [
                availability(
                    [t[field] for t in group]
                    if field == "wall_seconds"
                    else [r.get(field) for t in group for r in t["results"]]
                )
                for group in groups
            ]
            if not samples[0] or not samples[1]:
                # No request in the group exposed this metric; report it rather than fail.
                metrics.append(
                    {
                        "case": case,
                        "metric": field,
                        "available": False,
                        "note": "no measured request exposed this metric",
                    }
                )
                continue
            medians = [statistics.median(values) for values in samples]
            ratio = medians[1] / medians[0]
            metrics.append(
                {
                    "case": case,
                    "metric": field,
                    "available": True,
                    "samples": [len(values) for values in samples],
                    "baseline": medians[0],
                    "candidate": medians[1],
                    "ratio": ratio,
                    "passed": ratio <= max_ratio,
                }
            )
    performance = all(m.get("passed", True) for m in metrics)
    hardware = []
    for engine, report in (("baseline", baseline), ("candidate", candidate)):
        telemetry = report["telemetry"]
        # The window that covers only the running workload, so the mean is not diluted by the
        # idle gaps between groups; the span is kept for comparison.
        if not isinstance(telemetry.get("active_window"), dict):
            raise ValueError(f"{engine} report does not separate its active measurement window")
        if telemetry.get("active_process") is None:
            raise ValueError(f"{engine} report has no server-process telemetry")
    for case in sorted({key[0] for key in old}):
        sides = [
            [trial["gpu"] for key, trial in side.items() if key[0] == case] for side in (old, new)
        ]
        hardware.append(
            {
                "case": case,
                "baseline_utilization_gpu_mean": statistics.fmean(
                    entry["utilization_gpu_mean"] for entry in sides[0]
                ),
                "candidate_utilization_gpu_mean": statistics.fmean(
                    entry["utilization_gpu_mean"] for entry in sides[1]
                ),
                "baseline_utilization_memory_mean": statistics.fmean(
                    entry["utilization_memory_mean"] for entry in sides[0]
                ),
                "candidate_utilization_memory_mean": statistics.fmean(
                    entry["utilization_memory_mean"] for entry in sides[1]
                ),
                "baseline_sm_clock_mean_mhz": statistics.fmean(
                    entry["sm_clock_mean_mhz"] for entry in sides[0]
                ),
                "candidate_sm_clock_mean_mhz": statistics.fmean(
                    entry["sm_clock_mean_mhz"] for entry in sides[1]
                ),
            }
        )
    numeric = {
        "required": identical,
        "identical": token_mismatches == 0,
        "token_mismatches": token_mismatches,
    }
    passed = performance and (not identical or numeric["identical"])
    return {
        "passed": passed,
        "ratio_orientation": label,
        "max_latency_ratio": max_ratio,
        "token_mismatches": token_mismatches,
        "identical_tokens_required": identical,
        "performance": {"passed": performance, "metrics": metrics},
        "hardware": {
            "active_window": {
                engine: report["telemetry"]["active_window"]
                for engine, report in (("baseline", baseline), ("candidate", candidate))
            },
            "active_process": {
                engine: report["telemetry"]["active_process"]
                for engine, report in (("baseline", baseline), ("candidate", candidate))
            },
            "measured_span": {
                engine: report["telemetry"].get("measured_span")
                for engine, report in (("baseline", baseline), ("candidate", candidate))
            },
            "per_case": hardware,
        },
        "numeric": numeric,
        "quality": {
            "available": False,
            "note": "task quality is verified by its own gate, not here",
        },
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-ratio", type=float, default=1.1)
    parser.add_argument("--require-identical-tokens", action="store_true")
    parser.add_argument(
        "--require-identical-resources",
        action="store_true",
        help="require equal native startup execution geometry, scheduler and KV capacity",
    )
    parser.add_argument(
        "--allow-legacy-baseline",
        action="store_true",
        help="accept a frozen baseline whose release profile predates the effective-profile rule",
    )
    parser.add_argument(
        "--reference-engine",
        default="vllm",
        help="the side the target is measured against; the ratio is the other side over this one",
    )
    args = parser.parse_args()
    if not math.isfinite(args.max_ratio) or args.max_ratio <= 0:
        parser.error("--max-ratio must be positive and finite")
    try:
        result = compare(
            json.loads(args.baseline.read_text()),
            json.loads(args.candidate.read_text()),
            args.max_ratio,
            args.require_identical_tokens,
            args.reference_engine,
            args.allow_legacy_baseline,
            require_identical_resources=args.require_identical_resources,
        )
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
