"""Fixed-output serving must keep equal work and pass only declared server flags."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "fixed_work_serving", Path(__file__).with_name("fixed-work-serving.py")
)
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)


class FixedWorkTests(unittest.TestCase):
    def test_openai_completions_ignore_eos_and_native_payloads_pass_through(self):
        completion = {"prompt": [1, 2], "max_tokens": 4}
        self.assertTrue(fixed.fixed_work(completion)["ignore_eos"])
        self.assertNotIn("ignore_eos", completion)
        native = {"input": {"Sequence": {"tokens": [1]}}, "workload": {}}
        self.assertEqual(fixed.fixed_work(native), native)

    def test_zero_option_native_command_has_only_model_and_listen(self):
        command = fixed.server_command("native", Path("infer"), Path("model"), 7, [])
        self.assertEqual(command, ["infer", "model", "--listen", "127.0.0.1:7"])
        flagged = fixed.server_command(
            "native", Path("infer"), Path("m"), 7, ["--max-num-seqs", "16"]
        )
        self.assertEqual(flagged[-2:], ["--max-num-seqs", "16"])

    def test_groups_follow_repeat_order_with_dense_slots(self):
        rows = [
            {"case": "c", "repeat": 1, "tokens": [3]},
            {"case": "c", "repeat": 0, "tokens": [1]},
            {"case": "c", "repeat": 1, "tokens": [4]},
        ]
        grouped = fixed.groups(rows)
        self.assertEqual([len(group) for group in grouped], [1, 2])
        self.assertEqual([row["slot"] for row in grouped[1]], [0, 1])
        self.assertEqual(grouped[1][1]["tokens"], [4])

    def test_summary_excludes_warmup(self):
        report = {
            "completed": True,
            "groups": [
                {"warmup": True, "wall_seconds": 9.0},
                {"warmup": False, "wall_seconds": 3.0},
                {"warmup": False, "wall_seconds": 5.0},
            ],
        }
        self.assertEqual(fixed.summary(report)["median_wall"], 4.0)

    def test_equal_tokens_share_a_digest(self):
        self.assertEqual(fixed.digest([1, 2]), fixed.digest([1, 2]))
        self.assertNotEqual(fixed.digest([1, 2]), fixed.digest([2, 1]))


if __name__ == "__main__":
    unittest.main()
