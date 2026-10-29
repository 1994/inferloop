"""Contract tests for the image serving client without a GPU or external service."""

import asyncio
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "serve_image_load", Path(__file__).with_name("serve-image-load.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Handler(BaseHTTPRequestHandler):
    received: ClassVar[list] = []
    hosts: ClassVar[list] = []
    guard = threading.Lock()
    reject = False
    stream = False
    output_tokens = 3
    mode = None
    malformed_chunk = b""
    manifest_update = None
    second_arrived = threading.Event()

    def log_message(self, *_):
        pass

    def do_POST(self):
        if self.mode == "slow_read":
            with self.guard:
                self.received.append(None)
            time.sleep(0.7)
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.guard:
            self.received.append(request)
            self.hosts.append(self.headers.get("Host"))
            request_index = len(self.received)
            update = self.manifest_update
            Handler.manifest_update = None
        if update is not None:
            path, data = update
            path.write_bytes(data)
        if self.mode == "fail_inflight":
            if request_index == 1:
                self.second_arrived.wait(timeout=2)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"invalid json")
                return
            self.second_arrived.set()
            try:
                self.send_response(200)
                self.end_headers()
                for _ in range(40):
                    self.wfile.write(b"still active\n")
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                pass
            return
        if self.mode in ("heartbeat", "slow_header", "slow_chunk", "slow_body"):
            try:
                if self.mode == "slow_header":
                    self.connection.sendall(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                else:
                    self.send_response(200)
                    if self.mode == "slow_chunk":
                        self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                for _ in range(40):
                    data = b": heartbeat\n\n" if self.mode == "heartbeat" else b"1"
                    self.wfile.write(data)
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                pass
            return
        if self.mode == "oversize_stream":
            self.send_response(200)
            self.end_headers()
            try:
                self.wfile.write(b"x" * 2048)
                self.wfile.flush()
            except OSError:
                pass
            return
        if self.mode == "malformed_chunk":
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(self.malformed_chunk)
            self.wfile.flush()
            return
        if self.mode in ("paced_stream", "paced_chunked", "paced_frame"):
            first = {
                "id": "fixture-id",
                "model": "fixture",
                "choices": [{"index": 0, "delta": {"content": "first"}, "finish_reason": None}],
            }
            finish = {
                "id": "fixture-id",
                "model": "fixture",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            }
            usage = {
                "id": "fixture-id",
                "model": "fixture",
                "choices": [],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
            beginning = b"data: " + json.dumps(first).encode() + b"\n\n"
            rest = (
                b"".join(
                    b"data: " + json.dumps(item).encode() + b"\n\n" for item in (finish, usage)
                )
                + b"data: [DONE]\n\n"
            )
            self.send_response(200)
            if self.mode == "paced_chunked":
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", str(len(beginning) + len(rest)))
            self.end_headers()
            try:
                if self.mode == "paced_frame":
                    self.wfile.write(beginning[:-1])
                    self.wfile.flush()
                    time.sleep(0.25)
                    self.wfile.write(beginning[-1:])
                elif self.mode == "paced_chunked":
                    self.wfile.write(f"{len(beginning):x}\r\n".encode() + beginning + b"\r\n")
                else:
                    self.wfile.write(beginning)
                self.wfile.flush()
                time.sleep(0.25)
                if self.mode == "paced_chunked":
                    self.wfile.write(f"{len(rest):x}\r\n".encode() + rest + b"\r\n0\r\n\r\n")
                else:
                    self.wfile.write(rest)
                self.wfile.flush()
            except OSError:
                pass
            return
        if self.mode == "empty_stream":
            self.send_response(200)
            self.end_headers()
            for item in [
                {
                    "id": "fixture-id",
                    "model": "fixture",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                },
                {
                    "id": "fixture-id",
                    "model": "fixture",
                    "choices": [],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                },
            ]:
                self.wfile.write(b"data: " + json.dumps(item).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        if self.reject:
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b"busy")
            return
        self.send_response(200)
        if self.stream:
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for item in [
                {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "first"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": " second"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "length"}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 7, "completion_tokens": self.output_tokens},
                },
            ]:
                item["id"] = "fixture-id"
                item["model"] = "fixture"
                for choice in item["choices"]:
                    choice["index"] = 0
                self.wfile.write(b"data: " + json.dumps(item).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "id": "fixture-id",
                        "model": "fixture",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"content": "first second"},
                                "finish_reason": "length",
                            }
                        ],
                        "usage": {"prompt_tokens": 7, "completion_tokens": self.output_tokens},
                    }
                ).encode()
            )


class ImageLoadTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        entries = []
        for index in range(12):
            payload = b"test image " + str(index).encode()
            path = self.root / f"{index}.png"
            path.write_bytes(payload)
            entries.append({"file": path.name, "sha256": hashlib.sha256(payload).hexdigest()})
        (self.root / "manifest.json").write_text(
            json.dumps({"seed": 9, "items": {"small": entries}})
        )
        Handler.received = []
        Handler.hosts = []
        Handler.reject = False
        Handler.stream = False
        Handler.output_tokens = 3
        Handler.mode = None
        Handler.manifest_update = None
        Handler.second_arrived = threading.Event()
        # Avoid DNS reverse lookups in HTTPServer.server_bind on developer machines.
        with patch("socket.getfqdn", return_value="localhost"):
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.temp.cleanup()

    def args(self, *extra):
        return module.parse_args(
            [
                "--url",
                self.url,
                "--model",
                "fixture",
                "--engine-label",
                "test",
                "--fixtures",
                str(self.root),
                "--size",
                "small",
                "--concurrency",
                "2",
                "--samples",
                "4",
                "--warmup",
                "1",
                "--output",
                str(self.root / "out.json"),
                *extra,
            ]
        )

    def test_closed_loop_preserves_multiple_unique_images_and_resize_options(self):
        args = self.args(
            "--images-per-request",
            "2",
            "--processor-options",
            '{"do_resize":false}',
            "--require-output-tokens",
            "3",
        )
        report = module.run(args)
        self.assertEqual(report["counts"]["successes"], 4)
        self.assertEqual(report["counts"]["failures"], 0)
        self.assertEqual(report["latency_success"]["count"], 4)
        self.assertIsNone(report["first_visible_content"])
        self.assertEqual(len(Handler.received), 5)
        self.assertTrue(all(len(x["messages"][0]["content"]) == 3 for x in Handler.received))
        self.assertTrue(
            all(x["mm_processor_kwargs"] == {"do_resize": False} for x in Handler.received)
        )
        all_hashes = [
            i["sha256"] for r in report["warmups"] + report["samples"] for i in r["fixtures"]
        ]
        self.assertEqual(len(set(all_hashes)), 10)

    def test_streaming_records_first_visible_content_and_length(self):
        Handler.stream = True
        report = module.run(self.args("--stream", "--require-output-tokens", "3"))
        self.assertEqual(report["counts"]["successes"], 4)
        self.assertEqual(report["first_visible_content"]["count"], 4)
        self.assertTrue(all(x["text"] == "first second" for x in report["samples"]))
        self.assertTrue(all(x["usage"]["completion_tokens"] == 3 for x in report["samples"]))

    def test_warmup_failure_is_recorded_and_stops_measured_work(self):
        Handler.reject = True
        report = module.run(self.args("--warmup", "3"))
        self.assertEqual(report["warmups"][0]["status"], 429)
        self.assertEqual(report["warmups"][0]["outcome"], "http_error")
        self.assertEqual(report["counts"]["attempted"], 0)
        self.assertEqual(report["counts"]["unattempted"], 4)
        self.assertIsNone(report["successes_per_second"])
        self.assertEqual(len(Handler.received), 1)

    def test_warmup_timeout_stops_remaining_warmups(self):
        Handler.mode = "heartbeat"
        report = module.run(self.args("--warmup", "3", "--stream", "--timeout", "0.15"))
        self.assertEqual(len(report["warmups"]), 1)
        self.assertEqual(report["warmups"][0]["outcome"], "timeout")
        self.assertEqual(report["counts"]["attempted"], 0)
        self.assertEqual(report["counts"]["unattempted"], 4)
        self.assertEqual(len(Handler.received), 1)

    def test_measured_failure_cancels_inflight_and_does_not_start_more(self):
        Handler.mode = "fail_inflight"
        report = module.run(self.args("--warmup", "0", "--timeout", "2"))
        self.assertEqual(report["stop_reason"], "sample:invalid_response")
        self.assertEqual(report["counts"]["attempted"], 2)
        self.assertEqual(report["counts"]["unattempted"], 2)
        self.assertEqual(report["counts"]["by_outcome"], {"invalid_response": 1, "cancelled": 1})
        self.assertEqual(len(Handler.received), 2)
        self.assertTrue(all(sample["transport_close_confirmed"] for sample in report["samples"]))

    def test_capacity_rejections_remain_observed_without_retries(self):
        Handler.reject = True
        report = module.run(self.args("--warmup", "0"))
        self.assertIsNone(report["stop_reason"])
        self.assertEqual(report["counts"]["attempted"], 4)
        self.assertEqual(report["counts"]["unattempted"], 0)
        self.assertEqual(report["counts"]["by_http_status"], {"429": 4})
        self.assertEqual(len(Handler.received), 4)

    def test_missing_telemetry_keeps_request_results(self):
        report = module.run(self.args("--telemetry-file", str(self.root / "missing")))
        self.assertEqual(report["counts"]["successes"], 4)
        self.assertIn("FileNotFoundError", report["telemetry"]["error"])

    def test_monitor_summary_uses_only_samples_in_the_measurement_window(self):
        path = self.root / "telemetry.jsonl"
        records = [
            {
                "type": "metadata",
                "pid": 23,
                "interval_seconds": 1,
                "gpu_fields": ["index", "utilization.gpu"],
            },
            {
                "type": "sample",
                "unix_seconds": 1,
                "gpu": [["0", "10"]],
                "cpu_percent_one_core": {"23": 90},
                "processes": {"23": {"rss_bytes": 100}},
            },
            {
                "type": "sample",
                "unix_seconds": 2,
                "gpu": [["0", "75"]],
                "cpu_percent_one_core": {"23": 120},
                "processes": {"23": {"rss_bytes": 200}},
            },
        ]
        path.write_text("\n".join(json.dumps(record) for record in records))
        summary = module.telemetry_summary(path, 1500000000, 2500000000)
        self.assertEqual(summary["monitor_pid"], 23)
        self.assertEqual(summary["process_cpu"]["cpu_percent_one_core_mean"], 120)
        self.assertEqual(summary["gpu"]["utilization_gpu_mean"], 75)
        self.assertEqual(summary["process_tree_rss_peak_bytes"], 200)

    def test_output_length_mismatch_is_not_counted_as_success(self):
        report = module.run(self.args("--warmup", "0", "--require-output-tokens", "4"))
        self.assertEqual(report["counts"]["failures"], 2)
        self.assertGreaterEqual(report["counts"]["by_outcome"].get("output_length_mismatch", 0), 1)
        self.assertEqual(report["counts"]["unattempted"], 2)
        self.assertEqual(report["latency_failure"]["count"], 2)
        Handler.output_tokens = True
        report = module.run(self.args("--warmup", "0", "--require-output-tokens", "1"))
        self.assertGreaterEqual(report["counts"]["by_outcome"].get("output_length_mismatch", 0), 1)
        self.assertEqual(report["counts"]["unattempted"], 2)

    def test_comparison_preflight_rejects_unmatched_workloads_and_outputs(self):
        sample = {
            "fixtures": [{"file": "one.png", "sha256": "a" * 64}],
            "outcome": "success",
            "status": 200,
            "latency_ms": 1.5,
            "text": "answer",
            "finish_reason": "length",
            "response_model": "fixture",
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }
        reference = {
            "capture_contract": module.TEXT_CAPTURE_CONTRACT,
            "model": "fixture",
            "workload": {"prompt": "describe", "samples": 1, "warmup": 0},
            "samples": [sample],
            "warmups": [],
        }
        candidate = copy.deepcopy(reference)
        self.assertTrue(module.comparison_preflight(reference, candidate)["passed"])
        candidate["samples"][0]["usage"]["completion_tokens"] = 2
        self.assertFalse(module.comparison_preflight(reference, candidate)["passed"])
        candidate = copy.deepcopy(reference)
        candidate["samples"][0]["outcome"] = "http_error"
        self.assertFalse(module.comparison_preflight(reference, candidate)["passed"])
        candidate = copy.deepcopy(reference)
        candidate["samples"][0]["usage"].pop("prompt_tokens")
        self.assertFalse(module.comparison_preflight(reference, candidate)["passed"])
        candidate = copy.deepcopy(reference)
        candidate["samples"][0]["fixtures"][0]["sha256"] = "b" * 64
        self.assertFalse(module.comparison_preflight(reference, candidate)["passed"])
        candidate = copy.deepcopy(reference)
        candidate["workload"]["prompt"] = "different"
        self.assertFalse(module.comparison_preflight(reference, candidate)["passed"])
        candidate = copy.deepcopy(reference)
        candidate["model"] = "different"
        self.assertFalse(module.comparison_preflight(reference, candidate)["passed"])
        candidate = copy.deepcopy(reference)
        candidate["samples"][0]["response_model"] = "different"
        self.assertFalse(module.comparison_preflight(reference, candidate)["passed"])
        candidate = copy.deepcopy(reference)
        candidate["samples"][0]["status"] = 500
        self.assertFalse(module.comparison_preflight(reference, candidate)["passed"])
        candidate = copy.deepcopy(reference)
        candidate["samples"][0].pop("latency_ms")
        self.assertFalse(module.comparison_preflight(reference, candidate)["passed"])
        reference["workload"]["processor_options"] = {"do_resize": True}
        candidate = copy.deepcopy(reference)
        candidate["workload"]["processor_options"]["do_resize"] = 1
        self.assertFalse(module.comparison_preflight(reference, candidate)["passed"])
        candidate = copy.deepcopy(reference)
        candidate.pop("capture_contract")
        self.assertFalse(module.comparison_preflight(reference, candidate)["passed"])
        candidate = copy.deepcopy(reference)
        candidate["samples"][0]["text"] = "different answer"
        self.assertFalse(module.comparison_preflight(reference, candidate)["passed"])
        candidate = copy.deepcopy(reference)
        candidate["samples"][0]["usage"]["completion_tokens"] = True
        self.assertFalse(module.comparison_preflight(reference, candidate)["passed"])
        reference["workload"]["warmup"] = 1
        reference["warmups"] = [copy.deepcopy(sample)]
        candidate = copy.deepcopy(reference)
        candidate["warmups"][0]["fixtures"][0]["sha256"] = "b" * 64
        self.assertFalse(module.comparison_preflight(reference, candidate)["passed"])

    def test_reference_report_is_checked_against_full_requests(self):
        reference = module.run(self.args())
        path = self.root / "reference.json"
        path.write_text(json.dumps(reference))
        candidate = module.run(self.args("--reference-report", str(path)))
        self.assertTrue(candidate["comparison_preflight"]["passed"])
        Handler.output_tokens = 2
        candidate = module.run(self.args("--reference-report", str(path)))
        self.assertFalse(candidate["comparison_preflight"]["passed"])
        self.assertTrue(
            any(
                "completion_tokens" in issue
                for issue in candidate["comparison_preflight"]["issues"]
            )
        )

    def test_reference_is_read_before_starting_the_server_workload(self):
        with self.assertRaises(FileNotFoundError):
            module.run(self.args("--reference-report", str(self.root / "missing-reference")))
        self.assertEqual(Handler.received, [])

    def test_unsupported_nontext_output_cannot_be_compared_as_empty_text(self):
        output = {
            "id": "fixture-id",
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "message": {"content": None, "tool_calls": [{"id": "call-a"}]},
                    "finish_reason": "tool_calls",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "unsupported"):
            module.parse_complete(io.BytesIO(json.dumps(output).encode()))
        events = (
            b'data: {"id":"fixture-id","model":"fixture",'
            b'"choices":[{"index":0,"delta":{"reasoning_content":"hidden"}}]}\n\n'
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            module.parse_stream(io.BytesIO(events), 0)

    def test_comparison_preflight_rejects_invalid_reference_schema(self):
        invalid = {
            "capture_contract": module.TEXT_CAPTURE_CONTRACT,
            "model": "fixture",
            "workload": {"samples": 1, "warmup": 0},
            "warmups": [],
            "samples": [
                {
                    "fixtures": ["bogus"],
                    "outcome": "success",
                    "text": 123,
                    "finish_reason": "",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ],
        }
        self.assertFalse(module.comparison_preflight(invalid, copy.deepcopy(invalid))["passed"])

    def test_stream_must_be_one_continuous_text_choice(self):
        chunks = [
            {"id": "a", "model": "m", "choices": [{"index": 0, "delta": {"content": "sa"}}]},
            {"id": "b", "model": "m", "choices": [{"index": 0, "delta": {"content": "me"}}]},
            {
                "id": "a",
                "model": "m",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        events = b"".join(b"data: " + json.dumps(item).encode() + b"\n\n" for item in chunks)
        with self.assertRaises(ValueError):
            module.parse_stream(io.BytesIO(events + b"data: [DONE]\n\n"), 0)
        chunks[1]["id"] = "a"
        chunks[1]["choices"][0]["index"] = 1
        events = b"".join(b"data: " + json.dumps(item).encode() + b"\n\n" for item in chunks)
        with self.assertRaises(ValueError):
            module.parse_stream(io.BytesIO(events + b"data: [DONE]\n\n"), 0)
        chunks[1]["choices"][0]["index"] = 0
        chunks[1]["choices"][0]["delta"]["content"] = False
        events = b"".join(b"data: " + json.dumps(item).encode() + b"\n\n" for item in chunks)
        with self.assertRaises(ValueError):
            module.parse_stream(io.BytesIO(events + b"data: [DONE]\n\n"), 0)
        with self.assertRaises(ValueError):
            module.parse_complete(io.BytesIO(b'{"choices":[{"finish_reason":"stop"}]}'))

    def test_hash_mismatch_and_duplicate_fixtures_are_rejected(self):
        (self.root / "0.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            module.run(self.args())
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["items"]["small"][1] = manifest["items"]["small"][0]
        (self.root / "0.png").write_bytes(b"test image 0")
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "unique"):
            module.run(self.args())

    def test_manifest_digest_uses_bounded_bytes_parsed_before_requests(self):
        path = self.root / "manifest.json"
        original = path.read_bytes()
        expected = hashlib.sha256(original).hexdigest()
        for replacement in (b'{"seed": 100}', b"x" * (module.MAX_MANIFEST_BYTES + 1)):
            with self.subTest(bytes=len(replacement)):
                path.write_bytes(original)
                Handler.manifest_update = (path, replacement)
                report = module.run(self.args())
                self.assertEqual(report["counts"]["successes"], 4)
                self.assertEqual(report["workload"]["manifest_sha256"], expected)
                self.assertEqual(path.stat().st_size, len(replacement))

    def test_absolute_deadline_covers_slow_headers_heartbeats_and_chunk_framing(self):
        for mode in ("slow_header", "heartbeat", "slow_chunk", "slow_body"):
            with self.subTest(mode=mode):
                Handler.mode = mode
                started = time.monotonic()
                extra = ["--stream"] if mode != "slow_body" else []
                report = module.run(
                    self.args("--samples", "2", "--warmup", "0", "--timeout", "0.15", *extra)
                )
                self.assertLess(time.monotonic() - started, 1.0)
                self.assertEqual(report["counts"]["attempted"], 2)
                self.assertGreaterEqual(report["counts"]["by_outcome"].get("timeout", 0), 1)
                self.assertTrue(set(report["counts"]["by_outcome"]) <= {"timeout", "cancelled"})
                self.assertEqual(report["latency_failure"]["count"], 2)

    def test_stream_without_visible_content_fails_even_with_output_tokens(self):
        Handler.mode = "empty_stream"
        report = module.run(
            self.args("--samples", "2", "--warmup", "0", "--stream", "--require-output-tokens", "3")
        )
        self.assertGreaterEqual(report["counts"]["by_outcome"].get("invalid_response", 0), 1)
        self.assertTrue(set(report["counts"]["by_outcome"]) <= {"invalid_response", "cancelled"})
        self.assertIsNone(report["first_visible_content"])

    def test_first_content_time_does_not_wait_for_complete_content_length_body(self):
        for mode in ("paced_stream", "paced_chunked"):
            with self.subTest(mode=mode):
                Handler.mode = mode
                report = module.run(self.args("--samples", "2", "--warmup", "0", "--stream"))
                self.assertEqual(report["counts"]["successes"], 2)
                for sample in report["samples"]:
                    self.assertGreater(sample["latency_ms"] - sample["ttft_ms"], 150)

    def test_first_content_time_waits_for_complete_sse_event(self):
        Handler.mode = "paced_frame"
        report = module.run(self.args("--samples", "2", "--warmup", "0", "--stream"))
        self.assertEqual(report["counts"]["successes"], 2)
        for sample in report["samples"]:
            self.assertGreater(sample["ttft_ms"], 150)
            self.assertGreater(sample["latency_ms"] - sample["ttft_ms"], 150)

    def test_stream_event_framing_crlf_multiple_data_and_incomplete_eof(self):
        start = time.perf_counter()
        body = (
            b": keepalive\r\n\r\n"
            b'data: {"id":"a",\r\ndata: "model":"m","choices":[{"index":0,'
            b'"delta":{"content":"ok"},"finish_reason":"length"}]}\r\n\r\n'
            b"data: [DONE]\r\n\r\n"
        )
        result = module.parse_stream(io.BytesIO(body), start)
        self.assertEqual(result["text"], "ok")
        for bad in (body[:-1], body + b"data: extra\n", body + b"data: [DONE]\n\n"):
            with self.subTest(bad=bad[-30:]), self.assertRaisesRegex(ValueError, "stream"):
                module.parse_stream(io.BytesIO(bad), start)

    def test_stream_limit_rejects_long_unterminated_line_before_appending(self):
        Handler.mode = "oversize_stream"
        with patch.object(module, "MAX_RESPONSE_BYTES", 1024):
            capture = module.StreamCapture(time.perf_counter())
            with self.assertRaisesRegex(ValueError, "capture limit"):
                capture.feed(b"x" * 1025)
            self.assertEqual(capture.pending, bytearray())
            report = module.run(self.args("--samples", "2", "--warmup", "0", "--stream"))
        self.assertGreaterEqual(report["counts"]["by_outcome"].get("invalid_response", 0), 1)
        self.assertTrue(set(report["counts"]["by_outcome"]) <= {"invalid_response", "cancelled"})

    def test_rejects_invalid_chunk_size_or_line_ending(self):
        Handler.mode = "malformed_chunk"
        for chunk in (b"+1\r\n", b"0x1\r\n", b"1_0\r\n", b"1\n"):
            with self.subTest(chunk=chunk):
                Handler.malformed_chunk = chunk
                report = module.run(self.args("--samples", "2", "--warmup", "0"))
                self.assertGreaterEqual(
                    report["counts"]["by_outcome"].get("invalid_response", 0), 1
                )
                self.assertTrue(
                    set(report["counts"]["by_outcome"]) <= {"invalid_response", "cancelled"}
                )

    def test_fixture_file_and_aggregate_preparation_limits_fail_before_requests(self):
        with (
            patch.object(module, "MAX_FIXTURE_BYTES", 8),
            self.assertRaisesRegex(ValueError, "per-file client limit"),
        ):
            module.run(self.args())
        payload = b"a" * (300 * 1024)
        (self.root / "0.png").write_bytes(payload)
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["items"]["small"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "preparation limit"):
            module.run(self.args("--max-prepared-mib", "1"))
        self.assertEqual(Handler.received, [])

    def test_preparation_limit_default_custom_values_and_comparison_scope(self):
        reference = module.run(self.args())
        self.assertEqual(
            reference["client_preparation"]["conservative_preconstruction_limit"], 512 * 1024 * 1024
        )
        candidate = module.run(self.args("--max-prepared-mib", "1024"))
        self.assertEqual(
            candidate["client_preparation"]["conservative_preconstruction_limit"],
            1024 * 1024 * 1024,
        )
        self.assertIn("not a Python RSS limit", candidate["client_preparation"]["note"])
        self.assertTrue(module.comparison_preflight(reference, candidate)["passed"])
        for value in ("0", "-1", "abc"):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.args("--max-prepared-mib", value)

    def test_larger_preparation_limit_allows_previously_rejected_corpus(self):
        payload = b"a" * (300 * 1024)
        (self.root / "0.png").write_bytes(payload)
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["items"]["small"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "preparation limit"):
            module.run(self.args("--max-prepared-mib", "1"))
        self.assertEqual(Handler.received, [])
        report = module.run(self.args("--max-prepared-mib", "8"))
        self.assertEqual(report["counts"]["successes"], 4)
        self.assertEqual(
            report["client_preparation"]["conservative_preconstruction_limit"], 8 * 1024 * 1024
        )

    def test_slow_reader_aborts_before_starting_the_next_request(self):
        Handler.mode = "slow_read"
        entries = []
        for index in range(2):
            raw = bytes([index + 1]) * (8 * 1024 * 1024)
            path = self.root / f"large-{index}.png"
            path.write_bytes(raw)
            entries.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
        (self.root / "manifest.json").write_text(json.dumps({"items": {"large": entries}}))
        report = module.run(
            self.args(
                "--size",
                "large",
                "--samples",
                "2",
                "--concurrency",
                "1",
                "--warmup",
                "0",
                "--timeout",
                "0.15",
            )
        )
        self.assertEqual(report["counts"]["by_outcome"], {"timeout": 1})
        self.assertEqual(report["counts"]["unattempted"], 1)
        for sample in report["samples"]:
            self.assertTrue(sample["transport_close_confirmed"])
            self.assertLessEqual(
                sample["client_write_buffer_peak_bytes"], 2 * module.SEND_CHUNK_BYTES
            )
        self.assertEqual(len(Handler.received), 1)

    def test_cancellation_returns_active_failures_and_unattempted_counts(self):
        Handler.mode = "heartbeat"
        args = self.args("--warmup", "0", "--stream", "--timeout", "5")
        manifest, manifest_sha256, images = module.fixture_payloads(args)
        groups = [[image] for image in images]
        bodies = [module.request_body(args, group) for group in groups]
        url = args.url + "/v1/chat/completions"
        server = module.resolve_server(url)

        async def cancel_after_requests_arrive():
            task = asyncio.create_task(
                module.run_async(args, None, manifest, manifest_sha256, groups, bodies, url, server)
            )
            async with asyncio.timeout(2):
                while len(Handler.received) < 2:
                    await asyncio.sleep(0.01)
                task.cancel()
                return await task

        report = asyncio.run(cancel_after_requests_arrive())
        self.assertTrue(report["interrupted"])
        self.assertEqual(report["counts"]["attempted"], 2)
        self.assertEqual(report["counts"]["unattempted"], 2)
        self.assertEqual(report["counts"]["by_outcome"], {"cancelled": 2})

    def test_cancellation_during_warmup_keeps_failure_and_skips_samples(self):
        Handler.mode = "heartbeat"
        args = self.args("--warmup", "1", "--stream", "--timeout", "5")
        manifest, manifest_sha256, images = module.fixture_payloads(args)
        groups = [[image] for image in images]
        bodies = [module.request_body(args, group) for group in groups]
        url = args.url + "/v1/chat/completions"
        server = module.resolve_server(url)

        async def cancel_warmup():
            task = asyncio.create_task(
                module.run_async(args, None, manifest, manifest_sha256, groups, bodies, url, server)
            )
            async with asyncio.timeout(2):
                while not Handler.received:
                    await asyncio.sleep(0.01)
                task.cancel()
                return await task

        report = asyncio.run(cancel_warmup())
        self.assertTrue(report["interrupted"])
        self.assertEqual(report["warmups"][0]["outcome"], "cancelled")
        self.assertEqual(report["counts"]["attempted"], 0)
        self.assertEqual(report["counts"]["unattempted"], 4)

    def test_stream_comparison_requires_first_visible_content_for_every_sample(self):
        Handler.stream = True
        reference = module.run(self.args("--stream"))
        candidate = copy.deepcopy(reference)
        candidate["samples"][0]["ttft_ms"] = None
        result = module.comparison_preflight(reference, candidate)
        self.assertFalse(result["passed"])
        self.assertIn("sample_0:first_visible_content_missing_or_invalid", result["issues"])

    def test_keyboard_interrupt_saves_the_partial_report(self):
        Handler.mode = "heartbeat"
        path = self.root / "interrupted.json"
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("serve-image-load.py")),
                "--url",
                self.url,
                "--model",
                "fixture",
                "--engine-label",
                "test",
                "--fixtures",
                str(self.root),
                "--size",
                "small",
                "--concurrency",
                "2",
                "--samples",
                "4",
                "--warmup",
                "0",
                "--stream",
                "--timeout",
                "5",
                "--output",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 2
            while len(Handler.received) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(Handler.received), 2)
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=2)
            self.assertEqual(process.returncode, 1, (stdout, stderr))
            report = json.loads(path.read_text())
            self.assertTrue(report["interrupted"])
            self.assertEqual(report["counts"]["attempted"], 2)
            self.assertEqual(report["counts"]["unattempted"], 2)
            self.assertEqual(report["counts"]["by_outcome"], {"cancelled": 2})
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=2)

    def test_server_hostname_uses_explicit_numeric_address_without_dns(self):
        port = self.server.server_port
        args = self.args("--url", f"http://benchmark.example:{port}", "--connect-ip", "127.0.0.1")
        with patch("socket.getaddrinfo", side_effect=AssertionError("unexpected DNS call")):
            report = module.run(args)
        self.assertEqual(report["counts"]["successes"], 4)
        self.assertEqual(report["connection_ip"], "127.0.0.1")
        self.assertTrue(all(host == f"benchmark.example:{port}" for host in Handler.hosts))
        with self.assertRaisesRegex(ValueError, "numeric --connect-ip"):
            module.resolve_server("http://benchmark.example:80")
        target, addresses, tls = module.resolve_server("http://localhost:80")
        self.assertEqual(target.hostname, "localhost")
        self.assertEqual(addresses[0][4][0], "127.0.0.1")
        self.assertIsNone(tls)


if __name__ == "__main__":
    unittest.main()
