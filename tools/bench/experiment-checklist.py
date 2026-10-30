#!/usr/bin/env python3
"""Declare one experiment profile once, then verify every run and print the comparison table.

The baseline plan requires the two engines to be aligned on real work, quality conditions and
resource constraints, and it forbids reusing a script's old defaults: the checklist is the single
source for a profile, and a run has to prove it matched it.

A checklist is a JSON document with the following shape; every field is required because a
missing condition is the failure mode this exists to prevent:

    {
      "schema": 1,
      "profile_id": "27b-mtp2-nvfp4",
      "model": {"path": "/models/...", "files": {"config.json": "<sha256>", ...}},
      "numeric": {"weights": "...", "activation": "...", "kv": "..."},
      "quality": {"gate": "<reference to the quality check>", "required": false},
      "resources": {"gpu_memory_utilization": 0.88, "max_model_len": 262144,
                    "max_num_seqs": 16},
      "cache": {"prefix_cache": true},
      "workload": {"inputs_sha256": "...", "max_new_tokens": 64, "temperature": 0,
                   "mtp_depth": 2, "eos_tokens": [151645],
                   "matrix": {"short": {"concurrency": 1, "repeats": 3}, ...}},
      "engines": {"native": {}, "vllm": {"version": "0.31.0"}},
      "measurement": {"repeats": 3, "max_ratio": 1.1, "require_identical_tokens": false}
    }

Usage:
    python3 tools/bench/experiment-checklist.py --check profile.json reports/native.json
    python3 tools/bench/experiment-checklist.py --compare reports/native.json reports/vllm.json
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

# Every condition a profile declares. A missing one is reported rather than defaulted.
REQUIRED_SECTIONS = (
    "profile_id",
    "model",
    "numeric",
    "quality",
    "resources",
    "cache",
    "workload",
    "engines",
    "measurement",
)


def load(path):
    """Read and structurally validate a checklist."""
    return load_document(json.loads(Path(path).read_text()))


# A numeric contract that is only a dtype name answers nothing: the plan requires the weight
# handling, activation and accumulation rules, KV precision and sampling to be recorded, and names
# these three labels as the examples of what is not enough.
BARE_DTYPE_NAMES = {
    "bf16",
    "f16",
    "fp16",
    "fp32",
    "fp8",
    "f32",
    "int4",
    "int8",
    "mxfp4",
    "nvfp4",
    "w4a16",
    "w8a8",
}
NUMERIC_FIELDS = ("weights", "activation", "kv")


def require_numeric_contract(checklist):
    """Each numeric field has to describe a rule, not repeat a dtype name."""
    numeric = checklist.get("numeric")
    if not isinstance(numeric, dict):
        raise ValueError("checklist numeric contract must be a mapping")
    for field in NUMERIC_FIELDS:
        value = numeric.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"checklist numeric contract is missing {field}")
        if value.strip().lower() in BARE_DTYPE_NAMES:
            raise ValueError(
                f"checklist numeric {field} is only the dtype name {value.strip()!r}; record the "
                "weight handling, accumulation or state rule it stands for"
            )


# Keys a checkpoint uses to declare an MTP head, at the top level or under `text_config`, which is
# where the multimodal packages put the language tower's settings.
MTP_LAYER_KEYS = ("mtp_num_hidden_layers",)


def declared_mtp_layers(model_path):
    """How many MTP layers a checkpoint declares, or None when its configuration is unreadable.

    Zero is a real answer: it means the checkpoint ships no MTP head, so no speculative tier can be
    measured on it.
    """
    config = Path(model_path) / "config.json"
    if not config.is_file():
        return None
    document = json.loads(config.read_text())
    scopes = [document, document.get("text_config", {})]
    for scope in scopes:
        for key in MTP_LAYER_KEYS:
            if key in scope:
                return int(scope[key])
    return 0


def process_pair_orders(measurement):
    """Resolve the frozen per-pair order; old profiles retain alternating AB/BA."""
    pairs = measurement["fresh_process_pairs"]
    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 5:
        raise ValueError("fresh_process_pairs must be an integer of at least five")
    if "process_pair_orders" not in measurement:
        return [
            ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
            for index in range(pairs)
        ]
    orders = measurement["process_pair_orders"]
    if not isinstance(orders, list) or len(orders) != pairs:
        raise ValueError("process_pair_orders must declare one order for every frozen process pair")
    for order in orders:
        if order not in (["baseline", "candidate"], ["candidate", "baseline"]):
            raise ValueError(
                "each process_pair_orders entry must contain baseline and candidate once"
            )
    baseline_first = sum(order[0] == "baseline" for order in orders)
    if abs(2 * baseline_first - pairs) > 1:
        raise ValueError("process_pair_orders must be balanced: AB and BA counts may differ by one")
    return orders


def require_process_contract(checklist):
    """Validate an optional fresh-process A/B contract without embedding workload constants."""
    measurement = checklist["measurement"]
    if "require_identical_resources" in measurement and not isinstance(
        measurement["require_identical_resources"], bool
    ):
        raise ValueError("require_identical_resources must be a boolean")
    if "fresh_process_pairs" not in measurement:
        if "process_pair_orders" in measurement:
            raise ValueError("process_pair_orders requires fresh_process_pairs")
        return
    process_pair_orders(measurement)
    confidence = measurement.get("paired_confidence")
    if not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("paired_confidence must be between zero and one")
    matrix = checklist["workload"]["matrix"]
    primary = measurement.get("primary")
    if not isinstance(primary, dict):
        raise ValueError("process contract needs a primary metric")
    if primary.get("case") not in matrix:
        raise ValueError("primary case is absent from the workload matrix")
    for field in ("metric", "minimum_improvement_percent"):
        if field not in primary:
            raise ValueError(f"primary metric is missing {field}")
    if primary["metric"] not in ("wall_seconds", "ttft_seconds", "tpot_seconds"):
        raise ValueError("unsupported primary metric")
    if primary["minimum_improvement_percent"] <= 0:
        raise ValueError("minimum improvement must be positive")
    guards = measurement.get("guards")
    if not isinstance(guards, list):
        raise ValueError("process contract guards must be a list")
    for guard in guards:
        if guard.get("case") not in matrix:
            raise ValueError("guard case is absent from the workload matrix")
        if guard.get("metric") not in ("wall_seconds", "ttft_seconds", "tpot_seconds"):
            raise ValueError("unsupported guard metric")
        if guard.get("max_regression_percent", -1) < 0:
            raise ValueError("guard regression limit must be non-negative")


def load_document(checklist):
    """Structurally validate a checklist document."""
    if checklist.get("schema") != 1:
        raise ValueError("checklist schema must be 1")
    missing = [section for section in REQUIRED_SECTIONS if section not in checklist]
    if missing:
        raise ValueError(f"checklist is missing sections: {', '.join(missing)}")
    for field in ("path", "files"):
        if field not in checklist["model"]:
            raise ValueError(f"checklist model is missing {field}")
    for field in ("gpu_memory_utilization", "max_model_len", "max_num_seqs"):
        if field not in checklist["resources"]:
            raise ValueError(f"checklist resources are missing {field}")
    for field in ("prefix_cache",):
        if field not in checklist["cache"]:
            raise ValueError(f"checklist cache is missing {field}")
    for field in (
        "inputs_sha256",
        "max_new_tokens",
        "temperature",
        "mtp_depth",
        "eos_tokens",
        "matrix",
    ):
        if field not in checklist["workload"]:
            raise ValueError(f"checklist workload is missing {field}")
    require_numeric_contract(checklist)
    # Sampling that is not deterministic has to name the seed it drew from, or the two engines are
    # not running the same experiment however equal the temperature looks.
    if checklist["workload"]["temperature"] > 0 and not checklist["workload"].get("seed"):
        raise ValueError(
            "checklist samples at temperature > 0 without a seed, so the two engines cannot be "
            "aligned on the same draws"
        )
    # The baseline plan is explicit that a model without the algorithm must not be measured by
    # falling back to plain decode and calling the result that algorithm.
    depth = checklist["workload"]["mtp_depth"]
    if depth > 0:
        layers = declared_mtp_layers(checklist["model"]["path"])
        if layers == 0:
            raise ValueError(
                f"checklist asks for MTP depth {depth}, but {checklist['model']['path']} declares "
                "no MTP head, so this would measure a plain-decode fallback"
            )
    require_process_contract(checklist)
    return checklist


def _mismatch(mismatches, item, expected, actual):
    if expected != actual:
        mismatches.append(f"{item}: checklist declares {expected!r}, run has {actual!r}")


def check_run(checklist, report):
    """Return the ways a measured run failed to match its checklist.

    The report's recorded values are the ones the harness observed, including the effective
    configuration read back from the server, so this compares declared conditions against what
    actually happened rather than against the command line that was passed.
    """
    mismatches = []
    identity = report.get("identity") or {}
    engine = identity.get("engine")
    _mismatch(mismatches, "model path", checklist["model"]["path"], report.get("model"))
    files = (identity.get("model") or {}).get("files") or {}
    for name, digest in checklist["model"]["files"].items():
        _mismatch(mismatches, f"model artifact {name}", digest, files.get(name))
    _mismatch(
        mismatches,
        "gpu_memory_utilization",
        checklist["resources"]["gpu_memory_utilization"],
        report.get("gpu_memory_utilization"),
    )
    if "vision" in checklist["resources"]:
        _mismatch(
            mismatches,
            "vision availability",
            checklist["resources"]["vision"],
            report.get("vision_enabled"),
        )
    _mismatch(
        mismatches,
        "prefix cache",
        checklist["cache"]["prefix_cache"],
        report.get("prefix_cache_enabled"),
    )
    for item, field in (
        ("max_new_tokens", "max_new_tokens"),
        ("temperature", "temperature"),
        ("mtp_depth", "mtp_depth"),
        ("eos_tokens", "eos_tokens"),
    ):
        _mismatch(mismatches, item, checklist["workload"][field], report.get(field))
    _mismatch(
        mismatches,
        "workload inputs",
        checklist["workload"]["inputs_sha256"],
        report.get("inputs_sha256"),
    )
    _mismatch(mismatches, "matrix", checklist["workload"]["matrix"], report.get("matrix"))
    _mismatch(
        mismatches,
        "measured repeats",
        checklist["measurement"]["repeats"],
        next(iter(report.get("matrix", {}).values()), {}).get("repeats"),
    )
    # The context ceiling is read back from the server on native and declared on the reference
    # side; both have to match the profile, not just each other.
    limits = identity.get("limits") or {}
    if engine == "native":
        _mismatch(
            mismatches,
            "effective max model len",
            checklist["resources"]["max_model_len"],
            limits.get("effective_max_model_len"),
        )
    else:
        _mismatch(
            mismatches,
            "max_model_len",
            checklist["resources"]["max_model_len"],
            limits.get("max_model_len"),
        )
        _mismatch(
            mismatches,
            "max_num_seqs",
            checklist["resources"]["max_num_seqs"],
            limits.get("max_num_seqs"),
        )
        if "max_num_batched_tokens" in checklist["resources"]:
            _mismatch(
                mismatches,
                "max_num_batched_tokens",
                checklist["resources"]["max_num_batched_tokens"],
                limits.get("max_num_batched_tokens"),
            )
    expected_version = (checklist["engines"].get(engine) or {}).get("version")
    if expected_version is not None:
        _mismatch(
            mismatches,
            f"{engine} version",
            expected_version,
            identity.get("engine_version"),
        )
    return mismatches


def comparison_table(reports):
    """The per-item native/reference configuration table the baseline plan asks for."""
    by_engine = {report["identity"]["engine"]: report for report in reports}
    if len(by_engine) != len(reports):
        raise ValueError("the comparison table needs one report per engine")
    items = {
        "model": lambda r: r.get("model"),
        "model artifacts": lambda r: sorted(
            ((r.get("identity") or {}).get("model") or {}).get("files", {}).items()
        ),
        "context ceiling": lambda r: (
            ((r.get("identity") or {}).get("limits") or {}).get("effective_max_model_len")
            or ((r.get("identity") or {}).get("limits") or {}).get("max_model_len")
        ),
        "prefix cache": lambda r: r.get("prefix_cache_enabled"),
        "gpu_memory_utilization": lambda r: r.get("gpu_memory_utilization"),
        "mtp depth": lambda r: r.get("mtp_depth"),
        "output tokens": lambda r: r.get("max_new_tokens"),
        "temperature": lambda r: r.get("temperature"),
        "eos tokens": lambda r: r.get("eos_tokens"),
        "workload inputs": lambda r: r.get("inputs_sha256"),
        "matrix": lambda r: r.get("matrix"),
        "engine version": lambda r: (r.get("identity") or {}).get("engine_version"),
    }
    rows = []
    for item, extract in items.items():
        values = {engine: extract(report) for engine, report in by_engine.items()}
        rendered = {json.dumps(value, sort_keys=True) for value in values.values()}
        aligned = len(rendered) == 1
        rows.append({"item": item, "values": values, "aligned": aligned})
    return rows


def render(rows):
    engines = sorted({engine for row in rows for engine in row["values"]})
    width = max(len(row["item"]) for row in rows)
    lines = [
        f"{'item'.ljust(width)}  " + "  ".join(engine.ljust(24) for engine in engines) + "  aligned"
    ]
    for row in rows:
        cells = "  ".join(str(row["values"].get(engine, "-"))[:24].ljust(24) for engine in engines)
        lines.append(f"{row['item'].ljust(width)}  {cells}  {'yes' if row['aligned'] else 'NO'}")
    return "\n".join(lines)


def draft(arguments):
    """Build a checklist from a generated workload and the model it was tokenized for.

    The workload hash and the artifact fingerprints are computed here rather than typed in, so a
    profile cannot claim a condition nobody verified.
    """
    workload = Path(arguments.workload)
    rows = json.loads(workload.read_text())
    cases = {}
    for row in rows:
        entry = cases.setdefault(row["case"], {"concurrency": row["concurrency"], "repeats": 0})
        if entry["concurrency"] != row["concurrency"]:
            raise ValueError(f"{row['case']} declares inconsistent concurrency")
        entry["repeats"] = max(entry["repeats"], row["repeat"] + 1)
    model = Path(arguments.model)
    files = {
        name: hashlib.sha256((model / name).read_bytes()).hexdigest()
        for name in arguments.artifacts
        if (model / name).is_file()
    }
    if not files:
        raise ValueError(f"no known artifact found in {model}")
    spec = importlib.util.spec_from_file_location(
        "model_config", Path(__file__).with_name("model_config.py")
    )
    model_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_config)
    eos = model_config.eos_tokens(model)
    engines = {"native": {"version": None}} if arguments.engine == "native" else {}
    for engine in ("native", "vllm", "sglang"):
        if engine != arguments.engine:
            engines.setdefault(engine, {"version": None})
    engines[arguments.engine] = {"version": arguments.engine_version}
    return {
        "schema": 1,
        "profile_id": arguments.profile_id,
        "model": {"path": str(model), "files": files},
        "numeric": {
            "weights": arguments.weights,
            "activation": arguments.activation,
            "kv": arguments.kv,
        },
        "quality": {"gate": arguments.quality_gate, "required": arguments.quality_required},
        "resources": {
            "gpu_memory_utilization": arguments.gpu_memory_utilization,
            "max_model_len": arguments.max_model_len,
            "max_num_seqs": arguments.max_num_seqs,
        },
        "cache": {"prefix_cache": not arguments.no_prefix_cache},
        "workload": {
            "inputs_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            "max_new_tokens": arguments.tokens,
            "temperature": arguments.temperature,
            "mtp_depth": arguments.mtp,
            "eos_tokens": eos,
            "matrix": cases,
        },
        "engines": engines,
        "measurement": {
            "repeats": arguments.repeats,
            "max_ratio": arguments.max_ratio,
            "require_identical_tokens": arguments.require_identical_tokens,
        },
    }


ARTIFACTS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "chat_template.jinja",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, metavar="CHECKLIST", help="checklist to verify")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"))
    parser.add_argument("report", type=Path, nargs="?", help="report to check")
    parser.add_argument("--init", action="store_true", help="draft a profile from a workload")
    parser.add_argument("--out", type=Path, help="where --init writes the checklist")
    parser.add_argument("--profile-id", default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--workload", type=Path, default=None)
    parser.add_argument("--engine", default="native", choices=["native", "vllm", "sglang"])
    parser.add_argument("--engine-version", default=None)
    parser.add_argument("--weights", default="unspecified")
    parser.add_argument("--activation", default="unspecified")
    parser.add_argument("--kv", default="model default")
    parser.add_argument("--quality-gate", default="not declared")
    parser.add_argument("--quality-required", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--mtp", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-ratio", type=float, default=1.1)
    parser.add_argument("--require-identical-tokens", action="store_true")
    parser.add_argument("--no-prefix-cache", action="store_true")
    args = parser.parse_args()
    if args.init:
        if not (args.profile_id and args.model and args.workload and args.out):
            parser.error("--init needs --profile-id, --model, --workload and --out")
        args.artifacts = ARTIFACTS
        document = draft(args)
        load_document(document)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(document, indent=2) + "\n")
        print(f"{args.out}: profile {document['profile_id']}")
        return 0
    if args.compare is not None:
        reports = [json.loads(path.read_text()) for path in args.compare]
        rows = comparison_table(reports)
        print(render(rows))
        return 0 if all(row["aligned"] for row in rows) else 1
    if args.check is None or args.report is None:
        parser.error("provide --check CHECKLIST REPORT, or --compare A B")
    checklist = load(args.check)
    report = json.loads(args.report.read_text())
    mismatches = check_run(checklist, report)
    for mismatch in mismatches:
        print(mismatch, file=sys.stderr)
    if mismatches:
        print(f"{len(mismatches)} condition(s) differ from {args.check}", file=sys.stderr)
        return 1
    print(f"{args.report} matches {args.check} ({checklist['profile_id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
