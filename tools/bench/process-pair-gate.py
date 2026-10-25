#!/usr/bin/env python3
"""Judge fresh native A/B process pairs against one frozen experiment checklist."""

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def student_t_cdf(value, degrees):
    """Student-t CDF using only the Python standard library."""
    if degrees < 1:
        raise ValueError("Student-t degrees of freedom must be positive")
    if value == 0:
        return 0.5
    constant = math.exp(
        math.lgamma((degrees + 1) / 2)
        - math.lgamma(degrees / 2)
        - math.log(math.sqrt(math.pi * degrees))
    )

    def density(x):
        return constant * (1 + x * x / degrees) ** (-(degrees + 1) / 2)

    end = abs(value)
    intervals = 2048
    step = end / intervals
    weighted = density(0) + density(end)
    weighted += 4 * sum(density(step * index) for index in range(1, intervals, 2))
    weighted += 2 * sum(density(step * index) for index in range(2, intervals, 2))
    area = weighted * step / 3
    return 0.5 + math.copysign(area, value)


def student_t_quantile(probability, degrees):
    low, high = 0.0, 1.0
    while student_t_cdf(high, degrees) < probability:
        high *= 2
    for _ in range(64):
        middle = (low + high) / 2
        if student_t_cdf(middle, degrees) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def paired_interval(values, confidence):
    """Mean and two-sided Student-t interval for independent process-pair deltas."""
    if not values:
        raise ValueError("paired interval has no samples")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean, mean
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    radius = student_t_quantile((1 + confidence) / 2, len(values) - 1) * standard_error
    return mean, mean - radius, mean + radius


def metric_ratios(comparison):
    return {
        (row["case"], row["metric"]): row["ratio"]
        for row in comparison["metrics"]
        if row.get("available")
    }


def evaluate(checklist, checklist_sha256, pairs, load_report):
    """Evaluate all declared pairs; no sample is dropped or retried by this gate."""
    measurement = checklist["measurement"]
    contract = load_module(
        "experiment_checklist", Path(__file__).with_name("experiment-checklist.py")
    )
    orders = contract.process_pair_orders(measurement)
    required = len(orders)
    if len(pairs) != required:
        raise ValueError(f"expected exactly {required} frozen process pairs, found {len(pairs)}")
    compare = load_module("compare_results", Path(__file__).with_name("compare-results.py"))
    primary = measurement["primary"]
    primary_key = (primary["case"], primary["metric"])
    improvements = []
    regressions = {(guard["case"], guard["metric"]): [] for guard in measurement["guards"]}
    run_ids = set()
    evidence = []
    for index, pair in enumerate(pairs):
        expected = orders[index]
        if pair.get("order") != expected:
            raise ValueError(f"pair {index} must execute in the frozen process order: {expected}")
        reports = {side: load_report(pair[side]) for side in ("baseline", "candidate")}
        for side, report in reports.items():
            identity = report.get("identity") or {}
            if identity.get("engine") != "native":
                raise ValueError(f"pair {index} {side} is not a native fresh-process report")
            run_id = identity.get("run_id")
            if not run_id or run_id in run_ids:
                raise ValueError(f"pair {index} {side} has a missing or duplicate run id")
            run_ids.add(run_id)
            if (report.get("checklist") or {}).get("sha256") != checklist_sha256:
                raise ValueError(f"pair {index} {side} used a different experiment checklist")
        comparison = compare.compare(
            reports["baseline"],
            reports["candidate"],
            require_identical_resources=measurement.get("require_identical_resources", False),
        )
        ratios = metric_ratios(comparison)
        if primary_key not in ratios:
            raise ValueError(f"pair {index} has no primary metric {primary_key}")
        improvements.append(100 * (1 - ratios[primary_key]))
        for key in regressions:
            if key not in ratios:
                raise ValueError(f"pair {index} has no guard metric {key}")
            regressions[key].append(100 * (ratios[key] - 1))
        evidence.append({"pair": index, "order": pair["order"], "ratios": comparison["metrics"]})
    confidence = measurement["paired_confidence"]
    mean, lower, upper = paired_interval(improvements, confidence)
    primary_result = {
        "case": primary["case"],
        "metric": primary["metric"],
        "improvement_percent_mean": mean,
        "improvement_percent_lower95": lower,
        "improvement_percent_upper95": upper,
        "minimum_improvement_percent": primary["minimum_improvement_percent"],
        "passed": lower > primary["minimum_improvement_percent"],
    }
    guard_results = []
    for guard in measurement["guards"]:
        key = (guard["case"], guard["metric"])
        mean, lower, upper = paired_interval(regressions[key], confidence)
        guard_results.append(
            {
                "case": guard["case"],
                "metric": guard["metric"],
                "regression_percent_mean": mean,
                "regression_percent_lower95": lower,
                "regression_percent_upper95": upper,
                "maximum_regression_percent": guard["max_regression_percent"],
                "passed": upper <= guard["max_regression_percent"],
            }
        )
    passed = primary_result["passed"] and all(row["passed"] for row in guard_results)
    return {
        "schema": 1,
        "profile_id": checklist["profile_id"],
        "checklist_sha256": checklist_sha256,
        "fresh_process_pairs": len(pairs),
        "fresh_processes": 2 * len(pairs),
        "primary": primary_result,
        "guards": guard_results,
        "pairs": evidence,
        "passed": passed,
        "status": "passed" if passed else "not_demonstrated",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checklist", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path, help="JSON with schema and pairs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checklist_module = load_module(
        "experiment_checklist", Path(__file__).with_name("experiment-checklist.py")
    )
    checklist = checklist_module.load(args.checklist)
    checklist_sha256 = hashlib.sha256(args.checklist.read_bytes()).hexdigest()
    document = json.loads(args.pairs.read_text())
    if document.get("schema") != 1 or not isinstance(document.get("pairs"), list):
        raise ValueError("pairs manifest needs schema 1 and a pairs list")
    root = args.pairs.parent

    def load_report(name):
        return json.loads((root / name).read_text())

    result = evaluate(checklist, checklist_sha256, document["pairs"], load_report)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
