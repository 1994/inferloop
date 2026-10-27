"""Short snapshots must not issue release certificates or escape their deadline."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "iteration_gate", Path(__file__).with_name("iteration-gate.py")
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def serving(metrics=None):
    return {
        "readback": {
            "readback": {
                "execution_profile": {"draft_depth": 2, "attention_profile": "bf16"},
                "serving_plan": {
                    "width": 15,
                    "capacity": 8192,
                    "kv_tokens": 67776,
                    "prompt_width": 1024,
                    "quoted_bytes": 100,
                    "allocated_bytes": 90,
                    "envelope_bytes": 110,
                },
            }
        },
        "metrics": metrics
        or {
            name: {"wall_seconds": 2, "ttft_seconds": 1, "tpot_seconds": 0.01}
            for name in ["c16", "short1"]
        },
    }


class IterationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        fields = [
            "model",
            "executable",
            "baseline_executable",
            "suite",
            "baseline",
            "workload",
            "short_workload",
            "tuning_seed",
        ]
        values = {}
        for field in fields:
            path = self.root / field
            path.write_text("{}")
            values[field] = path
        self.args = SimpleNamespace(
            **values,
            output=self.root / "out",
            tokens=4,
            concurrency=16,
            startup_timeout=120,
            regression_percent=10,
        )
        self.scored = [{"correct": True} for _ in range(128)]
        self.report = {
            "completed": False,
            "collection_completed": True,
            "results": [{"correct": True} for _ in range(64)],
            "screening": [],
        }
        suite = {"samples": [{}] * 128, "generation": {"max_new_tokens": 8}, "eos_tokens": [99]}
        patches = [
            mock.patch.object(gate.harness, "require_protected_run"),
            mock.patch.object(
                gate.quality, "document", side_effect=[(suite, "suite"), ({}, "base")]
            ),
            mock.patch.object(gate.quality.scorer, "validate_suite"),
            mock.patch.object(gate.quality, "validate_baseline", return_value=self.scored),
            mock.patch.object(gate.harness, "model_identity", return_value={"files": {}}),
            mock.patch.object(gate.harness, "hardware_identity", return_value={"devices": ["gpu"]}),
            mock.patch.object(gate.harness, "eos_tokens", return_value=[99]),
            mock.patch.object(gate.harness, "binary_identity", return_value={"release_like": True}),
            mock.patch.object(gate, "performance", side_effect=[serving(), serving()]),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.collect = mock.patch.object(gate.quality, "collect", return_value=self.report).start()
        self.addCleanup(mock.patch.stopall)

    def run_gate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return gate.run(self.args, decode=lambda ids: "#### 2")

    def test_one_pair_and_64_tasks_return_both_metrics_without_release_acceptance(self):
        result = self.run_gate()
        self.assertTrue(result["iteration_accepted"])
        for key in ["release_accepted", "performance_accepted", "quality_accepted"]:
            self.assertFalse(result[key])
        self.assertEqual(result["quality"]["tasks"], 64)
        self.assertEqual(set(result["performance"]), {"c16", "short1"})
        self.assertEqual(self.collect.call_args.kwargs["sample_limit"], 64)
        self.assertEqual(result["time_budget_seconds"], 600)

    def test_gross_regression_stops_before_quality_server(self):
        gate.performance.side_effect = [
            serving(),
            serving(
                {
                    name: {"wall_seconds": 3, "ttft_seconds": 1, "tpot_seconds": 0.01}
                    for name in ["c16", "short1"]
                }
            ),
        ]
        with self.assertRaisesRegex(ValueError, "gross performance"):
            self.run_gate()
        self.collect.assert_not_called()
        result = json.loads((self.args.output / "summary.json").read_text())
        self.assertFalse(result["iteration_accepted"])

    def test_incomplete_quality_does_not_pass(self):
        self.report["results"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete iteration"):
            self.run_gate()
        result = json.loads((self.args.output / "summary.json").read_text())
        self.assertEqual(result["quality"]["tasks"], 63)
        self.assertFalse(result["iteration_accepted"])

    def test_input_mutation_invalidates_snapshot(self):
        def mutate(*args, **kwargs):
            self.args.workload.write_text("changed")
            return self.report

        self.collect.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "frozen input changed"):
            self.run_gate()

    def test_expired_budget_never_passes(self):
        with (
            mock.patch.object(gate.time, "monotonic", return_value=601),
            self.assertRaises(TimeoutError),
        ):
            gate.remaining(SimpleNamespace(deadline=600))

    def test_added_concurrency_diagnostics_preserve_comparable_execution(self):
        baseline, candidate = serving(), serving()
        profile = candidate["readback"]["readback"]["execution_profile"]
        profile["concurrency"] = {"memory_bound": 15, "roofline_bound": 52, "selected": 15}
        self.assertEqual(set(gate.compare(baseline, candidate, 10)), {"c16", "short1"})
        self.assertIn("concurrency", profile)
        profile["attention_profile"] = "fp8"
        with self.assertRaisesRegex(ValueError, "runtime execution profiles differ"):
            gate.compare(baseline, candidate, 10)

    def test_invalid_concurrency_diagnostics_cannot_hide_resource_changes(self):
        invalid = [
            {"memory_bound": 15, "roofline_bound": 52, "selected": 14},
            {"memory_bound": 14, "roofline_bound": 52, "selected": 15},
            {"memory_bound": 15, "roofline_bound": 14, "selected": 15},
            {"memory_bound": 15, "roofline_bound": 0, "selected": 15},
            {"memory_bound": 15, "roofline_bound": 52, "selected": True},
            {},
        ]
        for bounds in invalid:
            with self.subTest(bounds=bounds):
                candidate = serving()
                candidate["readback"]["readback"]["execution_profile"]["concurrency"] = bounds
                with self.assertRaisesRegex(ValueError, "concurrency"):
                    gate.compare(serving(), candidate, 10)
        candidate = serving()
        candidate["readback"]["readback"]["execution_profile"]["concurrency"] = {
            "memory_bound": 15,
            "roofline_bound": None,
            "selected": 15,
        }
        gate.compare(serving(), candidate, 10)

    def test_funded_resource_change_is_explicit_failure(self):
        candidate = serving()
        candidate["readback"]["readback"]["serving_plan"]["width"] = 14
        with self.assertRaisesRegex(ValueError, "funded serving resources differ"):
            gate.compare(serving(), candidate, 10)


class SafeRunTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.capture = self.root / "args.json"
        scripts = {
            "awk": "#!/bin/sh\necho 999999999\n",
            "flock": "#!/bin/sh\nexit 0\n",
            "systemd-run": (
                "#!/usr/bin/env python3\nimport json,os,sys\n"
                "open(os.environ['CAPTURE'],'w').write(json.dumps(sys.argv[1:]))\n"
            ),
        }
        for name, content in scripts.items():
            path = self.root / name
            path.write_text(content)
            path.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": str(self.root) + os.pathsep + os.environ["PATH"],
            "CAPTURE": str(self.capture),
        }
        self.script = Path(__file__).with_name("safe-run.sh")

    def invoke(self, *options):
        return subprocess.run(
            ["bash", str(self.script), *options, "true"],
            env=self.env,
            capture_output=True,
            text=True,
        )

    def test_ten_minute_budget_reserves_cleanup_for_the_whole_group(self):
        result = self.invoke("--memory-gib", "24", "--time-limit-seconds", "600")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.capture.read_text())
        self.assertIn("--property=RuntimeMaxSec=590s", args)
        self.assertIn("--property=TimeoutStopSec=10s", args)
        self.assertIn("--property=KillSignal=SIGINT", args)
        self.assertIn("--property=KillMode=control-group", args)

    def test_invalid_budget_never_launches_a_command(self):
        for value in ["0", "10", "1801", "9999999999999999999999", "bad", "0600"]:
            with self.subTest(value=value):
                self.assertEqual(self.invoke("--time-limit-seconds", value).returncode, 2)
                self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
