"""Contract tests for the agent replay client against a scripted Responses server."""

import importlib.util
import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "agent_replay", Path(__file__).with_name("agent-replay.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

TRACE = Path(__file__).with_name("agent-traces") / "coding-fix.json"


class Handler(BaseHTTPRequestHandler):
    """Answers every call with one tool call (or a message after `calls_before_message`)."""

    received: ClassVar[list] = []
    reused: ClassVar[list] = [0]
    calls_before_message: ClassVar[int] = 99
    reuse_per_call: ClassVar[int] = 0
    status: ClassVar[str] = "completed"

    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == module.OBSERVABILITY_ROUTE:
            body = {"metrics": {"prefix_tokens_reused": self.reused[0]}}
        else:
            body = {"data": [{"id": "fake-model"}]}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.received.append(payload)
        index = len(self.received)
        self.reused[0] += self.reuse_per_call
        if index > self.calls_before_message:
            item = {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }
            delta = "response.output_text.delta"
        else:
            item = {
                "type": "function_call",
                "id": f"fc_{index}",
                "call_id": f"model_{index}",
                "name": "read_file",
                "arguments": '{"path": "upload/ratelimit.py"}',
                "status": "completed",
            }
            delta = "response.function_call_arguments.delta"
        input_tokens = 100 * index
        response = {
            "id": f"resp_{index}",
            "status": self.status,
            "output": [item],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 10,
                "output_tokens_details": {"reasoning_tokens": 4},
            },
        }
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for event, data in (
            (
                "response.created",
                {"type": "response.created", "response": {**response, "status": "in_progress"}},
            ),
            (delta, {"type": delta, "delta": "x"}),
            ("response.completed", {"type": "response.completed", "response": response}),
        ):
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
        self.close_connection = True


class ReplayTest(unittest.TestCase):
    def setUp(self):
        Handler.received = []
        Handler.reused = [0]
        Handler.calls_before_message = 99
        Handler.reuse_per_call = 0
        Handler.status = "completed"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, args=(0.01,), daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.trace = json.loads(TRACE.read_text())

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def args(self, *extra):
        return module.parse_args(["--trace", str(TRACE), "--output", "unused.json", *extra])

    def test_generated_history_feeds_back_model_calls_with_rebound_outputs(self):
        result = module.replay(
            self.base, "fake-model", self.trace, self.args(), sleep=lambda _: None
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(Handler.received), len(self.trace["rounds"]))
        second = Handler.received[1]["input"]
        self.assertEqual(second[1]["call_id"], "model_1")
        self.assertEqual(second[1]["type"], "function_call")
        self.assertNotIn("status", second[1])
        self.assertEqual(second[2], {**self.trace["rounds"][1]["input"][0], "call_id": "model_1"})
        self.assertNotIn("previous_response_id", Handler.received[1])

    def test_chain_sends_only_new_items_after_the_first_round(self):
        module.replay(
            self.base, "fake-model", self.trace, self.args("--chain"), sleep=lambda _: None
        )
        first, second = Handler.received[:2]
        self.assertTrue(first["store"])
        self.assertNotIn("previous_response_id", first)
        self.assertEqual(second["previous_response_id"], "resp_1")
        self.assertEqual([item["type"] for item in second["input"]], ["function_call_output"])

    def test_truncated_output_ends_generated_but_not_recorded_replays(self):
        Handler.status = "incomplete"
        generated = module.replay(
            self.base, "fake-model", self.trace, self.args(), sleep=lambda _: None
        )
        self.assertEqual((generated["status"], len(generated["rounds"])), ("incomplete", 1))
        recorded = module.replay(
            self.base,
            "fake-model",
            self.trace,
            self.args("--history", "recorded"),
            sleep=lambda _: None,
        )
        self.assertEqual(recorded["status"], "completed")
        self.assertEqual(len(recorded["rounds"]), len(self.trace["rounds"]))

    def test_trace_diverges_when_the_model_stops_calling_tools(self):
        Handler.calls_before_message = 2
        result = module.replay(
            self.base, "fake-model", self.trace, self.args(), sleep=lambda _: None
        )
        self.assertEqual(result["status"], "diverged")
        self.assertEqual(len(result["rounds"]), 3)

    def test_recorded_history_replays_recorded_outputs(self):
        module.replay(
            self.base,
            "fake-model",
            self.trace,
            self.args("--history", "recorded"),
            sleep=lambda _: None,
        )
        second = Handler.received[1]["input"]
        self.assertEqual(
            second[1:], [*self.trace["rounds"][0]["output"], *self.trace["rounds"][1]["input"]]
        )

    def test_repeated_prefill_counts_context_that_was_not_reused(self):
        Handler.reuse_per_call = 30
        result = module.replay(
            self.base, "fake-model", self.trace, self.args(), sleep=lambda _: None
        )
        second = result["rounds"][1]
        # Round 2 sees 200 input tokens; 110 were the previous context, 30 were reused.
        self.assertEqual(second["new_input_tokens"], 90)
        self.assertEqual(second["prefill_tokens_recomputed"], 170)
        self.assertEqual(second["prefill_tokens_repeated"], 80)
        summary = module.summarize([result])
        self.assertAlmostEqual(summary["reasoning_share_of_output"], 0.4)

    def test_tool_latency_is_slept_and_reported(self):
        slept = []
        result = module.replay(self.base, "fake-model", self.trace, self.args(), sleep=slept.append)
        expected = [entry["tool_latency_ms"] / 1000 for entry in self.trace["rounds"]]
        self.assertEqual(slept, expected)
        self.assertEqual(
            result["rounds"][1]["tool_wait_ms"], self.trace["rounds"][1]["tool_latency_ms"]
        )

    def test_main_writes_the_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch.object(module.time, "sleep"), redirect_stdout(io.StringIO()):
                module.main(["--base", self.base, "--trace", str(TRACE), "--output", str(output)])
            report = json.loads(output.read_text())
        self.assertEqual(report["model"], "fake-model")
        self.assertEqual(report["summary"]["rounds"], len(self.trace["rounds"]))


class CopyTest(unittest.TestCase):
    def test_chain_rejects_recorded_history(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            module.parse_args(["--trace", "t", "--output", "o", "--chain", "--history", "recorded"])

    def test_counts_long_runs_found_in_context(self):
        context = "def take(self, amount, now):\n    self.refill(now)\n"
        output = 'x "def take(self, amount, now):" y'
        self.assertEqual(
            module.copyable_characters(output, context, 8), len("def take(self, amount, now):")
        )

    def test_ignores_runs_shorter_than_the_span(self):
        self.assertEqual(module.copyable_characters("abc", "abcdef", 4), 0)
        self.assertEqual(module.copyable_characters("abcdxxxx", "abcd", 5), 0)

    def test_rebind_rejects_missing_calls(self):
        recorded = [{"type": "function_call_output", "call_id": "c1", "output": "a"}]
        self.assertIsNone(module.rebind_outputs(recorded, []))


if __name__ == "__main__":
    unittest.main()
