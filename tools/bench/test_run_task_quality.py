"""Collection must reuse only complete aligned baselines and load each candidate once."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "run_task_quality", Path(__file__).with_name("run-task-quality.py")
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Process:
    pid = 12345

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def send_signal(self, value):
        self.returncode = 0

    def wait(self, timeout):
        return self.returncode


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.suite = {
            "profile_id": "fixture-quality",
            "numeric_contract": {"state": "unchanged F32"},
            "generation": {"max_new_tokens": 8},
            "eos_tokens": [99],
            "acceptance": {"noninferiority_margin_accuracy": 0.02, "confidence": 0.95},
            "samples": [
                {
                    "id": str(i),
                    "row_id": i,
                    "variant": "short" if i % 2 else "long",
                    "tokens": [100, i],
                    "reference_answer": "#### 2",
                }
                for i in range(256)
            ],
        }
        self.suite_path = self.root / "suite.json"
        self.suite_path.write_text(json.dumps(self.suite))
        self.suite_sha = hashlib.sha256(self.suite_path.read_bytes()).hexdigest()
        self.executable = self.root / "infer"
        self.executable.write_bytes(b"fixture executable")
        self.model = {"files": {"weights": "fixture-identity"}}
        self.hardware = {"devices": ["fixture-device"]}
        self.args = SimpleNamespace(
            suite=self.suite_path,
            model=self.root / "model",
            executable=self.executable,
            output=self.root / "new-run",
            baseline=None,
            concurrency=16,
            startup_timeout=30,
        )
        self.process = Process()
        h = runner.harness
        patches = [
            mock.patch.object(h, "require_protected_run"),
            mock.patch.object(h, "model_identity", return_value=self.model),
            mock.patch.object(h, "hardware_identity", return_value=self.hardware),
            mock.patch.object(h, "eos_tokens", return_value=[99]),
            mock.patch.object(
                h,
                "binary_identity",
                side_effect=lambda path: {
                    "sha256": h.sha256_file(path),
                    "release_like": True,
                },
            ),
            mock.patch.object(h, "git_identity", return_value={"revision": "fixture"}),
            mock.patch.object(h, "wait_ready", return_value=0.1),
            mock.patch.object(h, "runtime_readback", return_value={"readback": {}}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.popen = mock.patch.object(
            runner.subprocess, "Popen", return_value=self.process
        ).start()
        self.addCleanup(mock.patch.stopall)
        self.metrics = mock.patch.object(
            h, "read_metrics", return_value={"metrics": {"prefix_tokens_reused": 0}}
        ).start()
        self.groups = mock.patch.object(h, "measure_group", side_effect=self.group).start()

    @staticmethod
    def group(args, base, eos, counter, rows):
        return {
            "results": [
                {
                    "token_ids": [2, 99],
                    "finish_reason": "Eos",
                    "truncated": False,
                    "finished": {"measurement": {"successful": True}},
                }
                for _ in rows
            ]
        }

    def baseline(self):
        report = {
            "completed": True,
            "profile_id": self.suite["profile_id"],
            "suite_sha256": self.suite_sha,
            "model": self.model,
            "hardware": self.hardware,
            "numeric_contract": self.suite["numeric_contract"],
            "binary_sha256": "a" * 64,
            "results": [
                {
                    "id": row["id"],
                    "row_id": row["row_id"],
                    "variant": row["variant"],
                    "input_token_ids": row["tokens"],
                    "token_ids": [2, 99],
                    "text": "#### 2",
                    "correct": True,
                    "finish_reason": "Eos",
                    "truncated": False,
                    "prefix_tokens_reused": 0,
                }
                for row in self.suite["samples"]
            ],
        }
        self.args.baseline = self.root / "baseline.json"
        self.args.baseline.write_text(json.dumps(report))
        return report

    def run_collection(self, decode=lambda ids: "#### 2"):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run(self.args, decode)

    def test_complete_suite_starts_one_server_and_reuses_explicit_baseline(self):
        before = self.baseline()
        result = self.run_collection()
        self.assertTrue(result["quality_accepted"])
        self.assertTrue(result["evidence"]["baseline_reused"])
        self.assertEqual(self.popen.call_count, 1)
        self.assertEqual(self.groups.call_count, 16)
        self.assertTrue(self.popen.call_args.kwargs["start_new_session"])
        self.assertEqual(json.loads(self.args.baseline.read_text()), before)
        report = json.loads((self.args.output / "report.json").read_text())
        self.assertEqual(len(report["results"]), 256)
        self.assertEqual(len(report["batches"]), 16)
        self.assertEqual(len((self.args.output / "results.jsonl").read_text().splitlines()), 256)

    def test_collection_without_baseline_does_not_claim_quality_pass(self):
        result = self.run_collection()
        self.assertEqual(result["quality_status"], "not_evaluated")
        self.assertNotIn("quality_accepted", result)
        self.assertFalse((self.args.output / "gate.json").exists())

    def test_wrong_baseline_identity_fails_before_loading_the_model(self):
        original = self.baseline()
        for key in ["suite_sha256", "model", "hardware", "numeric_contract"]:
            report = copy.deepcopy(original)
            report[key] = "wrong"
            self.args.baseline.write_text(json.dumps(report))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_collection()
        self.popen.assert_not_called()

    def test_incomplete_or_forged_baseline_cannot_be_reused(self):
        original = self.baseline()
        for change in ["incomplete", "missing", "forged"]:
            report = copy.deepcopy(original)
            if change == "incomplete":
                report["completed"] = False
            elif change == "missing":
                report["results"].pop()
            else:
                report["quality_accepted"] = True
                report["results"][0]["correct"] = False
            self.args.baseline.write_text(json.dumps(report))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_collection()
        self.popen.assert_not_called()

    def test_known_collection_setting_mismatch_fails_before_loading(self):
        report = self.baseline()
        report["collection"] = {"concurrency": 8, "tokens": 8}
        self.args.baseline.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "settings differ"):
            self.run_collection()
        self.popen.assert_not_called()

    def test_prefix_reuse_is_measured_and_rejected(self):
        self.metrics.side_effect = [
            {"metrics": {"prefix_tokens_reused": 0}},
            {"metrics": {"prefix_tokens_reused": 1}},
        ]
        with self.assertRaisesRegex(ValueError, "prefix counter"):
            self.run_collection()
        report = json.loads((self.args.output / "report.json").read_text())
        self.assertFalse(report["completed"])
        self.assertEqual(report["results"], [])
        self.assertEqual(self.process.returncode, 0)

    def test_interrupted_later_batch_preserves_verified_rows_and_stops_server(self):
        self.groups.side_effect = [
            self.group(None, None, None, None, range(16)),
            RuntimeError("transport interrupted"),
        ]
        with self.assertRaisesRegex(RuntimeError, "transport"):
            self.run_collection()
        report = json.loads((self.args.output / "report.json").read_text())
        self.assertFalse(report["completed"])
        self.assertEqual(report["count"], 16)
        self.assertEqual(len((self.args.output / "results.jsonl").read_text().splitlines()), 16)
        self.assertTrue((self.args.output / "run.json").exists())
        self.assertEqual(self.process.returncode, 0)
        with self.assertRaises(ValueError):
            runner.scorer.score(self.suite, report, self.suite_sha)

    def test_changed_frozen_baseline_prevents_a_gate_result(self):
        self.baseline()

        def mutate(*args):
            self.args.baseline.write_text(self.args.baseline.read_text() + "\n")
            return self.group(*args)

        self.groups.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "baseline changed"):
            self.run_collection()
        self.assertFalse((self.args.output / "gate.json").exists())

    def test_actual_regression_keeps_original_noninferiority_failure(self):
        self.baseline()
        result = self.run_collection(lambda ids: "#### 3")
        self.assertFalse(result["quality_accepted"])
        self.assertEqual(len(result["regressions"]), 256)
        self.assertEqual(result["noninferiority_margin_accuracy"], 0.02)
        self.assertEqual(self.popen.call_count, 1)

    def test_existing_evidence_directory_is_not_overwritten(self):
        self.args.output.mkdir()
        sentinel = self.args.output / "sentinel"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_collection()
        self.assertEqual(sentinel.read_text(), "keep")
        self.popen.assert_not_called()

    def test_wrong_tokenizer_or_eos_fails_before_startup(self):
        for field, value in [
            ("tokenizer_files", {"tokenizer.json": "wrong"}),
            ("eos_tokens", [100]),
        ]:
            suite = copy.deepcopy(self.suite)
            suite[field] = value
            self.suite_path.write_text(json.dumps(suite))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_collection()
        self.popen.assert_not_called()

    def test_unsuccessful_native_measurement_is_not_journaled(self):
        self.groups.return_value = {
            "results": [
                {
                    "token_ids": [2],
                    "finish_reason": "Eos",
                    "truncated": False,
                    "finished": {"measurement": {"successful": False}},
                }
                for _ in range(16)
            ]
        }
        self.groups.side_effect = None
        with self.assertRaisesRegex(ValueError, "unsuccessful"):
            self.run_collection()
        self.assertEqual((self.args.output / "results.jsonl").read_text(), "")
        self.assertFalse(json.loads((self.args.output / "report.json").read_text())["completed"])

    def test_missing_device_identity_fails_before_startup(self):
        self.hardware["devices"] = []
        with self.assertRaisesRegex(ValueError, "visible CUDA"):
            self.run_collection()
        self.popen.assert_not_called()

    def test_non_greedy_contract_is_rejected_before_startup(self):
        self.suite["generation"]["temperature"] = 0.5
        self.suite_path.write_text(json.dumps(self.suite))
        with self.assertRaisesRegex(ValueError, "greedy"):
            self.run_collection()
        self.popen.assert_not_called()


class ShutdownTests(unittest.TestCase):
    def test_unexpected_server_exit_is_not_a_complete_collection(self):
        process = mock.Mock()
        process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "exited before shutdown"):
            runner.stop(process)
        process.send_signal.assert_not_called()

    def test_timeout_kills_only_owned_group_and_reports_failure(self):
        process = mock.Mock(pid=12345)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 30), 0]
        with mock.patch.object(runner.os, "killpg") as kill, self.assertRaises(RuntimeError):
            runner.stop(process)
        kill.assert_called_once_with(12345, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
