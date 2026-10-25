#!/usr/bin/env python3
"""Recompute bounded GSM8K-style task quality from complete, paired model outputs.

This checks answers independently of recorded scores. It does not certify kernel numerics,
model-wide capability, release identity, or serving performance.
"""

import argparse
import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import NormalDist

FINAL_ANSWER = re.compile(r"(?m)^\s*####\s*([-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*(?:\$|%)?\s*$")
FINISH_REASONS = {"Eos", "Length", "stop", "length"}


def numeric_answer(text):
    matches = FINAL_ANSWER.findall(text)
    if not matches:
        return None
    try:
        # Compare Decimal values directly: normalize() can round long numbers to its context.
        value = Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def paired_score(gains, losses, count, difference):
    """Tango score statistic for H0: candidate minus baseline accuracy equals `difference`.

    Discordant pairs carry the paired evidence; the variance uses the loss-cell probability
    restricted to the null, which keeps the test valid when few tasks change.
    """
    linear = -(gains + losses) + (2 * count - gains + losses) * difference
    constant = -losses * difference * (1 - difference)
    discriminant = max(0.0, linear * linear - 8 * count * constant)
    loss_probability = (math.sqrt(discriminant) - linear) / (4 * count)
    variance = count * (2 * loss_probability + difference - difference * difference)
    numerator = gains - losses - count * difference
    if variance <= 0:
        return math.copysign(math.inf, numerator) if numerator else 0.0
    return numerator / math.sqrt(variance)


def accuracy_change_lower_bound(gains, losses, count, confidence):
    """One-sided lower confidence bound on paired accuracy change, by inverting the score test.

    Gains and losses both count: numeric changes flip tasks in both directions, and only the
    net accuracy change is a quality loss.
    """
    if count < 1 or gains < 0 or losses < 0 or gains + losses > count:
        raise ValueError("invalid paired task count")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    critical = NormalDist().inv_cdf(confidence)
    lower, upper = -1.0, (gains - losses) / count
    if upper <= lower:
        return lower
    for _ in range(72):
        middle = (lower + upper) / 2
        if paired_score(gains, losses, count, middle) > critical:
            lower = middle
        else:
            upper = middle
    return lower


def noninferiority(gains, losses, count, margin, confidence):
    bound = accuracy_change_lower_bound(len(gains), len(losses), count, confidence)
    return {
        "requests": count,
        "gains": len(gains),
        "losses": len(losses),
        "accuracy_change": (len(gains) - len(losses)) / count,
        "accuracy_change_lower_bound": bound,
        "accepted": bound >= -margin,
    }


def validate_suite(suite):
    samples = suite["samples"]
    ids = [sample["id"] for sample in samples]
    if not samples or len(ids) != len(set(ids)):
        raise ValueError("suite needs nonempty, unique task IDs")
    for sample in samples:
        tokens = sample.get("tokens")
        if not tokens or any(
            type(token) is not int or not 0 <= token <= 0xFFFFFFFF for token in tokens
        ):
            raise ValueError("suite needs integer token inputs")
        if numeric_answer(sample["reference_answer"]) is None:
            raise ValueError("suite reference needs an explicit numeric final answer")
    acceptance = suite["acceptance"]
    margin = acceptance["noninferiority_margin_accuracy"]
    confidence = acceptance["confidence"]
    if not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("invalid accuracy margin")
    accuracy_change_lower_bound(0, 0, len(samples), confidence)
    return samples


def score(suite, report, suite_sha256):
    samples = validate_suite(suite)
    results = report.get("results", [])
    if report.get("completed") is not True or report.get("error"):
        raise ValueError("incomplete or failed task report")
    if report.get("suite_sha256") != suite_sha256:
        raise ValueError("task suite hash differs")
    if report.get("profile_id") != suite["profile_id"]:
        raise ValueError("quality profile differs")
    if [row["id"] for row in results] != [sample["id"] for sample in samples]:
        raise ValueError("missing, duplicate, or reordered task results")
    scored = []
    for sample, row in zip(samples, results, strict=True):
        if row.get("input_token_ids") != sample["tokens"]:
            raise ValueError("task input tokens differ")
        if row.get("variant") != sample["variant"] or row.get("row_id") != sample["row_id"]:
            raise ValueError("task metadata differs")
        if row.get("finish_reason") not in FINISH_REASONS or row.get("truncated") is not False:
            raise ValueError("task request did not finish successfully")
        if row.get("prefix_tokens_reused") != 0:
            raise ValueError("task request reused a prefix")
        finished = row.get("finished")
        if (
            isinstance(finished, dict)
            and (finished.get("measurement") or {}).get("successful") is not True
        ):
            raise ValueError("native task measurement was unsuccessful")
        expected = numeric_answer(sample["reference_answer"])
        predicted = numeric_answer(row["text"])
        correct = predicted is not None and predicted == expected
        if "correct" in row and row["correct"] is not correct:
            raise ValueError("recorded task score disagrees with raw text")
        scored.append(
            {
                "id": sample["id"],
                "variant": sample["variant"],
                "correct": correct,
                "predicted": str(predicted) if predicted is not None else None,
                "expected": str(expected),
                "format_missing": predicted is None,
                "length_limited": row["finish_reason"] in {"Length", "length"},
            }
        )
    return scored


def aggregate(rows):
    count = len(rows)
    correct = sum(row["correct"] for row in rows)
    return {
        "requests": count,
        "correct": correct,
        "accuracy": correct / count,
        "format_missing": sum(row["format_missing"] for row in rows),
        "length_limited": sum(row["length_limited"] for row in rows),
    }


def compare(suite, baseline, candidate, suite_sha256):
    left, right = [score(suite, report, suite_sha256) for report in (baseline, candidate)]
    for field in ("model", "hardware", "numeric_contract"):
        if not baseline.get(field) or baseline[field] != candidate.get(field):
            raise ValueError(f"quality identity differs or is missing: {field}")
    acceptance = suite["acceptance"]
    margin = acceptance["noninferiority_margin_accuracy"]
    confidence = acceptance["confidence"]

    def paired(variant=None):
        pairs = [
            (a, b)
            for a, b in zip(left, right, strict=True)
            if variant is None or a["variant"] == variant
        ]
        losses = [a["id"] for a, b in pairs if a["correct"] and not b["correct"]]
        gains = [a["id"] for a, b in pairs if not a["correct"] and b["correct"]]
        return losses, gains, noninferiority(gains, losses, len(pairs), margin, confidence)

    losses, gains, overall = paired()
    strata = {}
    for variant in sorted({row["variant"] for row in left}):
        strata[variant] = {
            "baseline": aggregate([r for r in left if r["variant"] == variant]),
            "candidate": aggregate([r for r in right if r["variant"] == variant]),
            "noninferiority": paired(variant)[2],
        }
    accepted = overall["accepted"] and all(s["noninferiority"]["accepted"] for s in strata.values())
    return {
        "schema": 2,
        "scope": "fixed arithmetic task subset with declared generation budget",
        "baseline": aggregate(left),
        "candidate": aggregate(right),
        "strata": strata,
        "regressions": losses,
        "gains": gains,
        "accuracy_change": overall["accuracy_change"],
        "accuracy_change_lower_bound": overall["accuracy_change_lower_bound"],
        "confidence": confidence,
        "noninferiority_margin_accuracy": margin,
        "method": (
            "paired accuracy non-inferiority, one-sided Tango score bound, overall and per variant"
        ),
        "statistical_assumption": (
            "independent paired Bernoulli task outcomes; no general-capability guarantee"
        ),
        "quality_status": "pass" if accepted else "not_demonstrated",
        "quality_accepted": accepted,
        "performance_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new evidence path")
    paths = {"suite": args.suite, "baseline": args.baseline, "candidate": args.candidate}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    inputs = {name: json.loads(data) for name, data in raw.items()}
    result = compare(inputs["suite"], inputs["baseline"], inputs["candidate"], hashes["suite"])
    result["evidence"] = {
        name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "evidence"}, indent=2))
    raise SystemExit(0 if result["quality_accepted"] else 1)


if __name__ == "__main__":
    main()
