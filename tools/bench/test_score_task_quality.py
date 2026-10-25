"""Task quality must follow raw answers and complete aligned inputs, not recorded totals."""

import copy
import importlib.util
import math
import random
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "task_quality", Path(__file__).with_name("score-task-quality.py")
)
quality = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(quality)


def fixtures(count=256):
    samples = [
        {
            "id": str(i),
            "row_id": i,
            "variant": "long" if i % 2 else "short",
            "tokens": [100, i],
            "reference_answer": "Compute one plus one.\n#### 2",
        }
        for i in range(count)
    ]
    suite = {
        "profile_id": "test-quality",
        "samples": samples,
        "acceptance": {"noninferiority_margin_accuracy": 0.02, "confidence": 0.95},
    }
    report = {
        "completed": True,
        "profile_id": suite["profile_id"],
        "suite_sha256": "fixture-sha",
        "model": {"files": {"weights": "fixture-weights"}},
        "hardware": {"devices": ["fixture-device"]},
        "numeric_contract": {"state": "F32"},
        "results": [
            {
                "id": row["id"],
                "row_id": row["row_id"],
                "variant": row["variant"],
                "input_token_ids": row["tokens"],
                "text": "#### 2",
                "correct": True,
                "finish_reason": "Eos",
                "truncated": False,
                "prefix_tokens_reused": 0,
            }
            for row in samples
        ],
    }
    return suite, report, copy.deepcopy(report)


class AnswerTests(unittest.TestCase):
    def test_decimal_currency_percentage_and_sign(self):
        for text, expected in [("#### 1,250$", "1250"), ("#### -2.50", "-2.5"), ("#### 70%", "70")]:
            self.assertEqual(quality.numeric_answer(text), quality.Decimal(expected))

    def test_prose_numbers_are_not_final_answers(self):
        for text in [
            "The answer is 12",
            "#### NaN",
            "#### infinity",
            "#### 2e3",
            "#### 12 dollars",
        ]:
            self.assertIsNone(quality.numeric_answer(text))

    def test_long_integers_do_not_round_to_the_decimal_context(self):
        a = quality.numeric_answer("#### 1234567890123456789012345678901")
        b = quality.numeric_answer("#### 1234567890123456789012345678902")
        self.assertNotEqual(a, b)


class ConfidenceTests(unittest.TestCase):
    def test_score_is_zero_at_the_observed_change(self):
        self.assertAlmostEqual(quality.paired_score(18, 15, 1024, 3 / 1024), 0, places=12)

    def test_restricted_loss_probability_maximizes_the_null_likelihood(self):
        gains, losses, count, difference = 5, 12, 200, -0.02

        def likelihood(loss):
            gain = loss + difference
            return (
                gains * math.log(gain)
                + losses * math.log(loss)
                + (count - gains - losses) * math.log(1 - gain - loss)
            )

        grid = [i / 100000 for i in range(2001, 40000)]
        best = max(grid, key=likelihood)
        score = quality.paired_score(gains, losses, count, difference)
        variance = (gains - losses - count * difference) ** 2 / score**2 / count
        self.assertAlmostEqual((variance - difference + difference**2) / 2, best, places=4)

    def test_large_counts_agree_with_the_wald_bound(self):
        gains, losses, count = 400, 500, 100000
        change = (gains - losses) / count
        wald = change - 1.6448536269514722 * math.sqrt(
            ((gains + losses) / count - change**2) / count
        )
        bound = quality.accuracy_change_lower_bound(gains, losses, count, 0.95)
        self.assertAlmostEqual(bound, wald, delta=2e-5)

    def test_symmetric_flips_are_noninferior_but_net_losses_are_not(self):
        # 16 losses and 16 gains of 1,024 is numeric flip noise around an unchanged score.
        self.assertGreater(quality.accuracy_change_lower_bound(16, 16, 1024, 0.95), -0.02)
        self.assertLess(quality.accuracy_change_lower_bound(0, 30, 1024, 0.95), -0.02)

    def test_more_evidence_tightens_the_bound(self):
        small = quality.accuracy_change_lower_bound(0, 0, 50, 0.95)
        large = quality.accuracy_change_lower_bound(0, 0, 1024, 0.95)
        self.assertLess(small, large)
        self.assertLess(large, 0)
        self.assertEqual(quality.accuracy_change_lower_bound(0, 10, 10, 0.95), -1)

    def test_bad_counts_and_confidence_are_rejected(self):
        for gains, losses, count, confidence in [
            (0, 0, 0, 0.95),
            (-1, 0, 2, 0.95),
            (0, -1, 2, 0.95),
            (2, 1, 2, 0.95),
            (0, 0, 2, 1),
            (0, 0, 2, math.nan),
        ]:
            with self.assertRaises(ValueError):
                quality.accuracy_change_lower_bound(gains, losses, count, confidence)


class ClusterAndPlanningTests(unittest.TestCase):
    def test_singleton_questions_have_no_design_effect(self):
        changes = {i: [change] for i, change in enumerate([1] * 9 + [-1] * 7 + [0] * 1000)}
        self.assertEqual(quality.design_effect(changes), 1)
        self.assertEqual(quality.design_effect({0: [0], 1: [0]}), 1)

    def test_shared_questions_widen_the_bound(self):
        # Both variants of a question flip together, so 16 questions carry 32 losses.
        changes = {i: [-1, -1] if i < 16 else [0, 0] for i in range(512)}
        effect = quality.design_effect(changes)
        self.assertGreater(effect, 1.9)
        independent = quality.accuracy_change_lower_bound(0, 32, 1024, 0.95)
        clustered = quality.accuracy_change_lower_bound(0, 32, 1024, 0.95, effect)
        self.assertLess(clustered, independent)

    def test_bad_design_effect_is_rejected(self):
        for effect in [0.5, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                quality.accuracy_change_lower_bound(0, 0, 10, 0.95, effect)

    def test_required_tasks_follow_the_normal_approximation(self):
        z = 1.6448536269514722 + 1.2815515655446004
        self.assertEqual(
            quality.required_tasks(0.03, 0.02, 0.95, 0.9), math.ceil(z * z * 0.03 / 0.0004)
        )
        self.assertGreater(
            quality.required_tasks(0.03, 0.01, 0.95, 0.9),
            quality.required_tasks(0.03, 0.02, 0.95, 0.9),
        )
        with self.assertRaises(ValueError):
            quality.required_tasks(0, 0.02, 0.95, 0.9)


class CalibrationTests(unittest.TestCase):
    """Seeded simulation of the decision rule at the observed GSM8K flip rate (3%)."""

    trials = 1000

    def accept_rate(self, change, count=1024, discordance=0.03, margin=0.02):
        rng = random.Random(20261025)
        gain, loss = (discordance + change) / 2, (discordance - change) / 2
        accepted = 0
        for _ in range(self.trials):
            draws = [rng.random() for _ in range(count)]
            gains = sum(u < gain for u in draws)
            losses = sum(gain <= u < gain + loss for u in draws)
            bound = quality.accuracy_change_lower_bound(gains, losses, count, 0.95)
            accepted += bound >= -margin
        return accepted / self.trials

    def test_false_acceptance_at_the_margin_stays_near_the_declared_rate(self):
        self.assertLess(self.accept_rate(-0.02), 0.08)

    def test_unchanged_candidates_usually_pass(self):
        self.assertGreater(self.accept_rate(0.0), 0.9)


class ReportTests(unittest.TestCase):
    def compare(self, suite, baseline, candidate):
        return quality.compare(suite, baseline, candidate, "fixture-sha")

    def test_identical_successes_pass_without_a_performance_claim(self):
        result = self.compare(*fixtures(1024))
        self.assertTrue(result["quality_accepted"])
        self.assertFalse(result["performance_accepted"])

    def test_partial_duplicate_and_reordered_reports_fail(self):
        for edit in [
            lambda r: r.update(completed=False),
            lambda r: r["results"].pop(),
            lambda r: r["results"].reverse(),
            lambda r: r["results"].append(r["results"][0]),
        ]:
            suite, baseline, candidate = fixtures()
            edit(candidate)
            with self.assertRaises(ValueError):
                self.compare(suite, baseline, candidate)

    def test_identity_input_and_reuse_mismatches_fail(self):
        for edit in [
            lambda r: r.update(suite_sha256="other"),
            lambda r: r.update(profile_id="other"),
            lambda r: r.update(model={}),
            lambda r: r.update(hardware={}),
            lambda r: r.update(numeric_contract={"state": "BF16"}),
            lambda r: r["results"][0].update(input_token_ids=[2]),
            lambda r: r["results"][0].update(prefix_tokens_reused=1),
            lambda r: r["results"][0].update(finish_reason="Cancelled"),
            lambda r: r["results"][0].update(truncated=True),
            lambda r: r["results"][0].update(finished={"measurement": {"successful": False}}),
        ]:
            suite, baseline, candidate = fixtures()
            edit(candidate)
            with self.assertRaises(ValueError):
                self.compare(suite, baseline, candidate)

    def test_tampered_recorded_score_is_detected(self):
        suite, baseline, candidate = fixtures()
        candidate["results"][0]["text"] = "#### 3"
        with self.assertRaises(ValueError):
            self.compare(suite, baseline, candidate)

    def test_capped_missing_answer_remains_in_the_denominator(self):
        suite, baseline, candidate = fixtures()
        candidate["results"][0].update(
            text="Still calculating 2", correct=False, finish_reason="Length"
        )
        result = self.compare(suite, baseline, candidate)
        self.assertEqual(result["candidate"]["requests"], 256)
        self.assertEqual(result["candidate"]["length_limited"], 1)
        self.assertEqual(result["candidate"]["format_missing"], 1)
        self.assertEqual(result["regressions"], ["0"])

    def test_balanced_flips_pass_with_their_counts_reported(self):
        suite, baseline, candidate = fixtures(1024)
        for i in range(8):
            baseline["results"][i].update(text="#### 3", correct=False)
            candidate["results"][100 + i].update(text="#### 3", correct=False)
        result = self.compare(suite, baseline, candidate)
        self.assertEqual(len(result["regressions"]), 8)
        self.assertEqual(len(result["gains"]), 8)
        self.assertEqual(result["accuracy_change"], 0)
        self.assertTrue(result["quality_accepted"])

    def test_stratum_noise_within_the_margin_passes(self):
        suite, baseline, candidate = fixtures(1024)
        # v10 zero-shot pattern: 12 losses and 10 gains in one 512-task variant.
        for i in range(0, 24, 2):
            candidate["results"][i].update(text="#### 3", correct=False)
        for i in range(24, 44, 2):
            baseline["results"][i].update(text="#### 3", correct=False)
        result = self.compare(suite, baseline, candidate)
        short = result["strata"]["short"]["noninferiority"]
        self.assertTrue(short["point_accepted"])
        self.assertTrue(result["strata_consistent"])
        self.assertTrue(result["quality_accepted"])

    def test_net_loss_beyond_the_margin_fails(self):
        suite, baseline, candidate = fixtures(1024)
        for i in range(40):
            candidate["results"][i].update(text="#### 3", correct=False)
        result = self.compare(suite, baseline, candidate)
        self.assertLess(result["accuracy_change_lower_bound"], -0.02)
        self.assertFalse(result["quality_accepted"])
        self.assertEqual(result["quality_status"], "not_demonstrated")

    def test_loss_concentrated_in_one_variant_fails(self):
        suite, baseline, candidate = fixtures(1024)
        # Twenty losses in "short" are hidden by twenty gains in "long" overall.
        for i in range(0, 40, 2):
            candidate["results"][i].update(text="#### 3", correct=False)
        for i in range(1, 41, 2):
            baseline["results"][i].update(text="#### 3", correct=False)
        result = self.compare(suite, baseline, candidate)
        self.assertEqual(result["accuracy_change"], 0)
        self.assertFalse(result["strata"]["short"]["noninferiority"]["point_accepted"])
        self.assertTrue(result["strata"]["long"]["noninferiority"]["point_accepted"])
        self.assertFalse(result["strata_consistent"])
        self.assertFalse(result["quality_accepted"])


if __name__ == "__main__":
    unittest.main()
