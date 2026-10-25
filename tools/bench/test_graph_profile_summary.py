"""Graph profile summaries must aggregate every replay and keep graphs apart."""

import importlib.util
import json
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "graph_profile_summary", Path(__file__).with_name("graph-profile-summary.py")
)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def test_replays_and_operations_accumulate_per_graph(self):
        lines = [
            json.dumps({"graph": "prefill", "total_ms": 10.0, "by_op": {"linear": 7.0}}),
            json.dumps(
                {"graph": "prefill", "total_ms": 6.0, "by_op": {"linear": 4.0, "norm": 1.0}}
            ),
            "",
            json.dumps({"graph": "slot_verify", "total_ms": 3.0, "by_op": {"attention": 2.0}}),
        ]
        graphs = summary.summarize(lines)
        self.assertEqual(graphs["prefill"]["replays"], 2)
        self.assertEqual(graphs["prefill"]["total_ms"], 16.0)
        self.assertEqual(graphs["prefill"]["ops"]["linear"], 11.0)
        self.assertEqual(graphs["slot_verify"]["ops"]["attention"], 2.0)

    def test_render_orders_graphs_by_time(self):
        text = summary.render(
            summary.summarize(
                [
                    json.dumps({"graph": "small", "total_ms": 1.0, "by_op": {"add": 1.0}}),
                    json.dumps({"graph": "large", "total_ms": 9.0, "by_op": {"linear": 9.0}}),
                ]
            ),
            top=5,
        )
        self.assertLess(text.index("large:"), text.index("small:"))
        self.assertIn("all graphs 0.010 s", text)


if __name__ == "__main__":
    unittest.main()
