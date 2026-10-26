"""Release acceptance never runs full quality on failed or mismatched performance."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "release_gate", Path(__file__).with_name("release-gate.py")
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        checklist, pairs = root / "checklist.json", root / "pairs.json"
        checklist.write_text("{}")
        pairs.write_text(json.dumps({"schema": 1, "pairs": [{"candidate": "candidate.json"}]}))
        self.report_path = root / "candidate.json"
        self.report_path.write_text(json.dumps({"binary_sha256": "same"}))
        out = root / "quality"
        out.mkdir()
        self.args = SimpleNamespace(
            performance=[(checklist, pairs)], executable=root / "infer", output=out
        )
        self.quality = SimpleNamespace(
            harness=SimpleNamespace(sha256_file=mock.Mock(return_value="same")),
            run=mock.Mock(return_value={"quality_accepted": True}),
        )
        self.pairs_gate = SimpleNamespace(evaluate=mock.Mock(return_value={"passed": True}))
        modules = {
            "release_quality": self.quality,
            "release_pairs": self.pairs_gate,
            "release_contract": SimpleNamespace(load=mock.Mock(return_value={})),
        }
        patch = mock.patch.object(release, "module", side_effect=lambda name, _: modules[name])
        patch.start()
        self.addCleanup(patch.stop)

    def test_failed_performance_stops_before_quality(self):
        self.pairs_gate.evaluate.return_value = {"passed": False}
        with self.assertRaisesRegex(ValueError, "performance not demonstrated"):
            release.run(self.args)
        self.quality.run.assert_not_called()

    def test_different_candidate_stops_before_quality(self):
        self.report_path.write_text(json.dumps({"binary_sha256": "different"}))
        with self.assertRaisesRegex(ValueError, "different candidate"):
            release.run(self.args)
        self.quality.run.assert_not_called()

    def test_release_requires_the_complete_quality_decision(self):
        self.quality.run.return_value = {"quality_accepted": False}
        self.assertFalse(release.run(self.args)["release_accepted"])
        self.quality.run.assert_called_once_with(self.args)

    def test_passed_evidence_runs_original_full_collector(self):
        self.assertTrue(release.run(self.args)["release_accepted"])
        self.quality.run.assert_called_once_with(self.args)


if __name__ == "__main__":
    unittest.main()
