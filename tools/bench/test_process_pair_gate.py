"""Fresh-process statistics and identity checks for the native A/B gate."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "process_pair_gate", Path(__file__).with_name("process-pair-gate.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

fixture_spec = importlib.util.spec_from_file_location(
    "compare_fixture", Path(__file__).with_name("test_compare_results.py")
)
fixture = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture)

PROFILE_HASH = "profile-hash"


def profile():
    return {
        "profile_id": "c16-test",
        "workload": {"matrix": {"c16": {"concurrency": 16, "repeats": 3}}},
        "measurement": {
            "fresh_process_pairs": 5,
            "paired_confidence": 0.95,
            "max_ratio": 1.1,
            "primary": {
                "case": "c16",
                "metric": "wall_seconds",
                "minimum_improvement_percent": 2,
            },
            "guards": [{"case": "c16", "metric": "tpot_seconds", "max_regression_percent": 2}],
        },
    }


def report(run_id, wall=1.0, tpot=1.0):
    result = fixture.report(case="c16", concurrency=16)
    result["identity"]["run_id"] = run_id
    result["checklist"]["sha256"] = PROFILE_HASH
    for trial in result["trials"]:
        trial["wall_seconds"] *= wall
        for row in trial["results"]:
            row["tpot_seconds"] *= tpot
    return result


def pairs(wall=None, tpot=None, count=5):
    wall = wall or [0.95] * count
    tpot = tpot or [1.0] * count
    result = []
    for index in range(count):
        order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
        result.append(
            {
                "order": order,
                "baseline": f"a{index}",
                "candidate": f"c{index}",
            }
        )
    return result, {
        name: report(
            name,
            wall[index] if name.startswith("c") else 1.0,
            tpot[index] if name.startswith("c") else 1.0,
        )
        for index in range(count)
        for name in (f"a{index}", f"c{index}")
    }


def evaluate(manifest, reports):
    return module.evaluate(profile(), PROFILE_HASH, manifest, reports.__getitem__)


class StatisticsTest(unittest.TestCase):
    def test_student_t_quantile_matches_reference_values(self):
        self.assertAlmostEqual(module.student_t_quantile(0.975, 1), 12.7062047, places=5)
        self.assertAlmostEqual(module.student_t_quantile(0.975, 4), 2.7764451, places=5)
        self.assertAlmostEqual(module.student_t_quantile(0.975, 5), 2.5705818, places=5)

    def test_five_fresh_pairs_can_pass_the_primary_and_guard(self):
        manifest, reports = pairs()
        result = evaluate(manifest, reports)
        self.assertTrue(result["passed"])
        self.assertEqual(result["fresh_processes"], 10)
        self.assertGreater(result["primary"]["improvement_percent_lower95"], 2)

    def test_a_small_mean_gain_does_not_pass_when_its_interval_crosses_two_percent(self):
        manifest, reports = pairs(wall=[0.95, 0.95, 0.95, 0.95, 1.05])
        result = evaluate(manifest, reports)
        self.assertFalse(result["passed"])
        self.assertFalse(result["primary"]["passed"])

    def test_guard_interval_cannot_hide_a_large_tpot_regression(self):
        manifest, reports = pairs(tpot=[1.01, 1.01, 1.01, 1.01, 1.10])
        result = evaluate(manifest, reports)
        self.assertFalse(result["passed"])
        self.assertFalse(result["guards"][0]["passed"])

    def test_pair_count_order_and_identity_are_enforced(self):
        manifest, reports = pairs()
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            evaluate(manifest[:4], reports)
        manifest, reports = pairs()
        manifest[1]["order"] = ["baseline", "candidate"]
        with self.assertRaisesRegex(ValueError, "frozen process order"):
            evaluate(manifest, reports)
        manifest, reports = pairs()
        reports["c0"]["identity"]["run_id"] = "a0"
        with self.assertRaisesRegex(ValueError, "duplicate run id"):
            evaluate(manifest, reports)

    def test_frozen_non_alternating_order_is_used(self):
        manifest, reports = pairs(count=6)
        orders = [
            ["baseline", "candidate"],
            ["candidate", "baseline"],
            ["candidate", "baseline"],
            ["baseline", "candidate"],
            ["baseline", "candidate"],
            ["candidate", "baseline"],
        ]
        contract = profile()
        contract["measurement"].update(fresh_process_pairs=6, process_pair_orders=orders)
        for pair, order in zip(manifest, orders, strict=True):
            pair["order"] = order
        result = module.evaluate(contract, PROFILE_HASH, manifest, reports.__getitem__)
        self.assertTrue(result["passed"])
        self.assertEqual([pair["order"] for pair in result["pairs"]], orders)
        manifest[2]["order"] = ["baseline", "candidate"]
        with self.assertRaisesRegex(ValueError, "frozen process order"):
            module.evaluate(contract, PROFILE_HASH, manifest, reports.__getitem__)

    def test_extra_pairs_cannot_extend_the_frozen_sample(self):
        manifest, reports = pairs(count=6)
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            evaluate(manifest, reports)

    def test_frozen_resource_contract_reaches_pair_comparison(self):
        manifest, reports = pairs()
        contract = profile()
        contract["measurement"]["require_identical_resources"] = True
        for item in reports.values():
            item["config_readback"] = fixture.native_resources()
        self.assertTrue(
            module.evaluate(contract, PROFILE_HASH, manifest, reports.__getitem__)["passed"]
        )
        reports["c0"]["config_readback"]["readback"]["kv_cache"]["pool_bytes"] *= 2
        with self.assertRaisesRegex(ValueError, "unaligned native resource.*pool_bytes"):
            module.evaluate(contract, PROFILE_HASH, manifest, reports.__getitem__)

    def test_a_different_checklist_is_rejected(self):
        manifest, reports = pairs()
        reports["c0"]["checklist"]["sha256"] = "other-profile"
        with self.assertRaisesRegex(ValueError, "different experiment checklist"):
            evaluate(manifest, reports)


if __name__ == "__main__":
    unittest.main()
