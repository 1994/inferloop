"""The paired refactor check must read tuning logs the way the server writes them."""

import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "arch_verify", Path(__file__).with_name("arch-verify.py")
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

LOG = "\n".join(
    (
        "\x1b[2m2026-11-10T18:48:03Z\x1b[0m  INFO infer::tuning: "
        "attention from table-cache shape=4x128",
        "2026-11-10T18:48:03Z  INFO infer::tuning: attention from table-cache shape=8x128",
        "2026-11-10T18:48:03Z  INFO infer::tuning: projection selected from_table=true key=abc",
        "2026-11-10T18:48:03Z  WARN infer::tuning: recurrence fallback rows=4",
        '2026-11-10T18:48:03Z  INFO infer::tuning: norm chosen source="measured" key=x',
        "2026-11-10T18:48:03Z  INFO infer::serve: listening",
    )
)


def run(sha="a", measured=0, hits=None, bad=None, errors=0):
    return {
        "greedy_sha": sha,
        "measured": measured,
        "hits": hits or {"attention from": 1},
        "bad": bad or {},
        "throughput": {1: {"tok_s": 1.0, "errors": errors}},
    }


class TuningLogTest(unittest.TestCase):
    def test_hits_group_by_message_without_key_fields_or_colour(self):
        hits, bad = module.tuning_hits(LOG)
        self.assertEqual(
            hits, {"attention from table-cache": 2, "projection selected from_table=true": 1}
        )
        self.assertEqual(bad, {"recurrence fallback": 1})

    def test_only_measurements_count_as_measured(self):
        self.assertEqual(module.measured(LOG), 1)


class VerdictTest(unittest.TestCase):
    def test_equivalent_runs_have_no_differences(self):
        self.assertEqual(module.verdict(run(), run()), [])

    def test_each_kind_of_difference_is_reported(self):
        reasons = module.verdict(
            run(sha="b", measured=2, bad={"x": 1}, errors=1, hits={"other": 1}), run()
        )
        self.assertEqual(len(reasons), 5)
        self.assertIn("greedy output differs", reasons)

    def test_the_reused_token_counter_is_found_wherever_it_nests(self):
        document = {"metrics": {"kv": [{"block_size": 32}, {"reused_tokens": 4096}]}}
        self.assertEqual(module.reused_tokens(document), 4096)
        self.assertIsNone(module.reused_tokens({"metrics": {}}))

    def test_a_configuration_names_its_extra_serve_flags(self):
        self.assertEqual(
            module.parse_config("mtp=/m/27B --num-speculative-tokens 2"),
            ("mtp", "/m/27B", ["--num-speculative-tokens", "2"]),
        )


if __name__ == "__main__":
    unittest.main()
