"""Contract tests for the portable serving harness without a GPU."""

import importlib.util
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "serve_compare", Path(__file__).with_name("serve-compare.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

workload_path = Path(__file__).with_name("serve-workloads.py")
workload_spec = importlib.util.spec_from_file_location("serve_workloads", workload_path)
workloads = importlib.util.module_from_spec(workload_spec)
workload_spec.loader.exec_module(workloads)


def write(rows):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(rows, handle)
        return Path(handle.name)


# The harness reads the declared concurrency from the workload, exactly as serve-workloads.py
# writes it, so the fixtures must carry it too.
CONCURRENCY = {"short": 1, "long": 1, "batch4": 4, "hot_long": 1}


def complete_rows(repeats=3):
    return [
        {
            "case": case,
            "repeat": repeat,
            "slot": slot,
            "concurrency": CONCURRENCY[case],
            "tokens": [1, 2, 3],
        }
        for case in module.CASES
        for repeat in range(-1, repeats)
        for slot in range(CONCURRENCY[case])
    ]


class ServingPlanReadbackTest(unittest.TestCase):
    @staticmethod
    def planned(width=7, tokens=35072, quote=4493453156):
        return (
            "INFO infer::executor: planned pooled CUDA serving storage "
            f"planned_width={width} prompt_width=1024 quoted_bytes={quote} "
            f"capacity=8192 kv_tokens={tokens}\n"
        )

    @staticmethod
    def accepted(width=7, quote=4493453156):
        return (
            "INFO infer::executor: CUDA slot pool captured within serving envelope "
            f"width={width} capacity=8192 quoted_bytes={quote} "
            "allocated_bytes=4184529176 envelope_bytes=4493738323\n"
        )

    def test_last_server_capture_after_replan_is_the_effective_plan(self):
        text = (
            "INFO infer::load: device budget 26.7 GiB\n"
            + self.planned()
            + self.accepted()
            + "INFO infer::load: device budget 26.7 GiB\n"
            + self.planned()
            + "WARN pooled CUDA capture exceeded budget; replanning before publication\n"
            + self.planned(4, 20512, 3384849456)
            + self.accepted(4, 3384849456)
        )
        self.assertEqual(
            module.serving_plan_readback(text),
            {
                "source": "native server log",
                "width": 4,
                "capacity": 8192,
                "kv_tokens": 20512,
                "prompt_width": 1024,
                "quoted_bytes": 3384849456,
                "allocated_bytes": 4184529176,
                "envelope_bytes": 4493738323,
            },
        )

    def test_previous_probe_capture_is_not_the_final_server_plan(self):
        text = self.planned() + self.accepted() + "INFO infer::load: device budget 26.7 GiB\n"
        self.assertIsNone(module.serving_plan_readback(text))
        self.assertIsNone(module.serving_plan_readback(text + self.planned()))

    def test_an_unaccepted_or_unmatched_plan_is_unavailable(self):
        for text in (
            self.planned(),
            self.accepted(),
            self.planned() + self.accepted(4),
            self.planned() + self.accepted(7, 1234),
            self.planned() + self.accepted().replace("allocated_bytes=4184529176 ", ""),
            self.planned()
            + self.accepted().replace("allocated_bytes=4184529176", "allocated_bytes=4493738324"),
            self.planned()
            + self.accepted()
            + "WARN CUDA slot pool disabled; reclaiming cached prefixes failed\n",
        ):
            with self.subTest(text=text):
                self.assertIsNone(module.serving_plan_readback(text))

    def test_colored_tracing_and_activity_logs_preserve_accepted_plan(self):
        text = "\x1b[32m" + self.planned() + self.accepted() + "\x1b[0m"
        text += "INFO infer::http: request path=runtime status=200\n"
        self.assertEqual(module.serving_plan_readback(text)["width"], 7)


class LoadWorkloadTest(unittest.TestCase):
    def test_complete_workload_passes(self):
        path = write(complete_rows())
        rows = module.load_workload(path, 3)
        self.assertEqual(len(rows), (1 + 1 + 4 + 1) * 4)
        self.assertEqual(module.workload_matrix(rows, 3)["batch4"]["concurrency"], 4)
        path.unlink()

    def test_a_row_without_declared_concurrency_is_rejected(self):
        rows = complete_rows()
        del rows[0]["concurrency"]
        path = write(rows)
        with self.assertRaises(SystemExit):
            module.load_workload(path, 3)
        path.unlink()

    def test_a_single_slot_batch4_group_is_rejected(self):
        # Four concurrent requests are the point of the case; one slot is not that case.
        rows = [row for row in complete_rows() if not (row["case"] == "batch4" and row["slot"] > 0)]
        path = write(rows)
        with self.assertRaises(SystemExit):
            module.load_workload(path, 3)
        path.unlink()

    def test_inconsistent_concurrency_within_a_case_is_rejected(self):
        rows = complete_rows()
        for row in rows:
            if row["case"] == "short" and row["repeat"] == 1:
                row["concurrency"] = 2
        path = write(rows)
        with self.assertRaises(SystemExit):
            module.load_workload(path, 3)
        path.unlink()

    def test_a_case_missing_one_slot_of_a_group_is_rejected(self):
        rows = [
            row
            for row in complete_rows()
            if not (row["case"] == "batch4" and row["repeat"] == 2 and row["slot"] == 3)
        ]
        path = write(rows)
        with self.assertRaises(SystemExit):
            module.load_workload(path, 3)
        path.unlink()

    def test_missing_repeat_is_rejected(self):
        rows = [row for row in complete_rows() if row["repeat"] != 1]
        path = write(rows)
        with self.assertRaises(SystemExit):
            module.load_workload(path, 3)
        path.unlink()

    def test_empty_tokens_are_rejected(self):
        rows = complete_rows()
        rows[0]["tokens"] = []
        path = write(rows)
        with self.assertRaises(SystemExit):
            module.load_workload(path, 3)
        path.unlink()

    def test_input_cap_is_enforced(self):
        path = write(complete_rows())
        with self.assertRaises(SystemExit):
            module.load_workload(path, 3, max_input_tokens=2)
        path.unlink()


class WorkloadMatrixTest(unittest.TestCase):
    def test_long_prompt_is_a_superset_of_short(self):
        short = workloads.content("short", 0, 0)
        long = workloads.content("long", 0, 0)
        body = workloads.BODY
        self.assertEqual(short.count(body), 1)
        self.assertEqual(long.count(body), workloads.CASES["long"][0])
        self.assertIn(body, short)
        self.assertIn(body, long)

    def test_cases_declare_their_concurrency(self):
        for case, (copies, concurrency) in workloads.CASES.items():
            self.assertGreaterEqual(copies, 1, case)
            self.assertGreaterEqual(concurrency, 1, case)
        self.assertEqual(workloads.CASES["batch4"][1], 4)

    def test_hot_prompt_is_longer_than_the_long_case(self):
        hot = workloads.hot_content()
        long = workloads.content("long", 0, 0)
        self.assertGreater(len(hot), len(long))


class CommandTest(unittest.TestCase):
    def arguments(self, **overrides):
        values = {
            "engine": "native",
            "model": Path("/models/x"),
            "executable": Path("/bin/infer"),
            "mtp": 2,
            "gpu_memory_utilization": 0.88,
            "prefix_cache": True,
            "max_model_len": 8192,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 2048,
            "extra": [],
        }
        values.update(overrides)
        return type("Args", (), values)()

    def test_native_command_is_model_and_engine_independent(self):
        command = module.build_command(self.arguments(), 1234)
        self.assertIn("--num-speculative-tokens", command)
        self.assertIn("2", command)
        self.assertNotIn("--enable-prefix-caching", command)

    def test_native_receives_the_same_serving_limits_as_the_reference(self):
        # Otherwise the comparison would measure two different service constraints.
        command = module.build_command(self.arguments(), 1234)
        for flag, value in (("--max-model-len", "8192"), ("--max-num-seqs", "16")):
            self.assertIn(flag, command, flag)
            self.assertEqual(command[command.index(flag) + 1], value, flag)

    def test_vllm_command_enables_speculation_and_prefix_cache(self):
        arguments = self.arguments(engine="vllm", executable=Path("/venv/bin/python"))
        command = module.build_command(arguments, 1234)
        self.assertIn("--enable-prefix-caching", command)
        self.assertIn("--speculative-config", command)

    def test_sglang_command_uses_nextn(self):
        arguments = self.arguments(engine="sglang", executable=Path("/venv/bin/python"))
        command = module.build_command(arguments, 1234)
        self.assertIn("--speculative-algorithm", command)
        self.assertIn("NEXTN", command)

    def test_vision_is_explicit_instead_of_silently_disabled(self):
        for engine in ("native", "vllm"):
            with self.subTest(engine=engine):
                text_only = module.build_command(self.arguments(engine=engine), 1234)
                self.assertIn("--limit-mm-per-prompt", text_only)
                vision = module.build_command(self.arguments(engine=engine, vision=True), 1234)
                self.assertNotIn("--limit-mm-per-prompt", vision)

    def test_vllm_command_disables_prefix_caching_when_asked(self):
        arguments = self.arguments(
            engine="vllm", executable=Path("/venv/bin/python"), prefix_cache=False
        )
        command = module.build_command(arguments, 1234)
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertNotIn("--enable-prefix-caching", command)

    def test_native_cannot_claim_a_disabled_cache(self):
        arguments = self.arguments(prefix_cache=False)
        with self.assertRaises(RuntimeError):
            module.require_cache_switch(arguments)
        module.require_cache_switch(self.arguments(prefix_cache=True))


class ResourcePreflightTest(unittest.TestCase):
    def arguments(self):
        return SimpleNamespace(tokens=256, max_model_len=4352, max_num_seqs=16)

    def row(self, **overrides):
        row = {
            "case": "c16",
            "repeat": 0,
            "slot": 0,
            "concurrency": 16,
            "tokens": [1] * 4096,
        }
        row.update(overrides)
        return row

    def test_c16_exact_context_and_custom_case_pass_preflight(self):
        rows = [self.row()]
        module.validate_workload_resources(self.arguments(), rows)
        self.assertEqual(module.workload_matrix(rows, 3)["c16"]["concurrency"], 16)

    def test_impossible_workload_is_rejected_before_server_startup(self):
        with self.assertRaisesRegex(RuntimeError, "needs 4353 context tokens"):
            module.validate_workload_resources(self.arguments(), [self.row(tokens=[1] * 4097)])
        with self.assertRaisesRegex(RuntimeError, "concurrency 17"):
            module.validate_workload_resources(self.arguments(), [self.row(concurrency=17)])

    def test_reference_thread_defaults_are_bounded_and_explicit_values_survive(self):
        with mock.patch.dict(module.os.environ, {}, clear=True):
            environment = module.process_environment("vllm")
            self.assertEqual(environment["OMP_NUM_THREADS"], "1")
            self.assertEqual(environment["TOKENIZERS_PARALLELISM"], "false")
            self.assertNotIn("OMP_NUM_THREADS", module.process_environment("native"))
        with mock.patch.dict(module.os.environ, {"OMP_NUM_THREADS": "8"}, clear=True):
            self.assertEqual(module.process_environment("vllm")["OMP_NUM_THREADS"], "8")


def elf(section_names):
    """A minimal ELF64 image whose section table lists exactly the given names."""
    strings = b"\0"
    offsets = {}
    for name in section_names:
        offsets[name] = len(strings)
        strings += name.encode() + b"\0"
    shstrtab = len(section_names) + 1
    count = shstrtab + 1
    header = bytearray(64)
    header[0:4] = b"\x7fELF"
    header[4] = 2  # 64-bit
    header[5] = 1  # little endian
    header[6] = 1  # version
    strings_offset = 64 + count * 64
    header[0x28:0x30] = (64).to_bytes(8, "little")
    header[0x34:0x36] = (64).to_bytes(2, "little")
    header[0x3A:0x3C] = (64).to_bytes(2, "little")
    header[0x3C:0x3E] = count.to_bytes(2, "little")
    header[0x3E:0x40] = shstrtab.to_bytes(2, "little")
    sections = bytearray(count * 64)
    for index, name in enumerate(section_names, start=1):
        base = index * 64
        sections[base : base + 4] = offsets[name].to_bytes(4, "little")
    base = shstrtab * 64
    sections[base + 0x18 : base + 0x20] = strings_offset.to_bytes(8, "little")
    sections[base + 0x20 : base + 0x28] = len(strings).to_bytes(8, "little")
    return bytes(header) + bytes(sections) + strings


class ProtectedRunTest(unittest.TestCase):
    def test_measuring_outside_the_protected_scope_is_refused(self):
        # An unprotected job can swap, and the review found the previous numbers had been taken
        # outside the wrapper, so the harness refuses instead of recording them.
        with mock.patch.object(module, "Path") as path:
            path.return_value.with_name.return_value = Path("/nonexistent/check_limits.py")
            with self.assertRaises((RuntimeError, OSError)):
                module.require_protected_run()


class IdentityTest(unittest.TestCase):
    def test_the_section_table_decides_whether_a_build_is_release(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            release.write_bytes(elf([".text", ".rodata"]))
            debug = Path(directory) / "debug"
            debug.write_bytes(elf([".text", ".debug_info", ".debug_str"]))
            self.assertEqual(module.binary_identity(release)["image"], "elf")
            self.assertTrue(module.binary_identity(release)["release_like"])
            self.assertEqual(module.binary_identity(release)["debug_sections"], [])
            self.assertFalse(module.binary_identity(debug)["release_like"])
            self.assertIn(".debug_info", module.binary_identity(debug)["debug_sections"])
            self.assertEqual(
                module.binary_identity(release)["sha256"],
                module.sha256_bytes(release.read_bytes()),
            )

    def test_the_section_names_are_read_rather_than_searched_for(self):
        # A release image whose read-only data merely mentions the names must still be release:
        # std's backtrace symbolizer embeds them, and a byte scan would misread that.
        payload = elf([".text"]) + b".debug_info in a string table"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release"
            path.write_bytes(payload)
            self.assertTrue(module.binary_identity(path)["release_like"])
            self.assertEqual(module.binary_identity(path)["debug_sections"], [])

    def test_an_image_whose_format_is_unknown_is_unverifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opaque"
            path.write_bytes(b"not an executable image")
            identity = module.binary_identity(path)
            self.assertEqual(identity["image"], "unknown")
            self.assertIsNone(identity["release_like"])

    def test_model_identity_fingerprints_the_artifacts_that_matter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "tokenizer.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"weights")
            identity = module.model_identity(root)
            self.assertIn("config.json", identity["files"])
            self.assertIn("tokenizer.json", identity["files"])
            self.assertNotIn("golden.json", identity["files"])
            self.assertEqual(identity["shards"], {"model.safetensors": 7})
            # Two packages that differ only in weights are not the same artifact.
            (root / "model.safetensors").write_bytes(b"other")
            self.assertNotEqual(identity["shards"], module.model_identity(root)["shards"])


class GroupStartTest(unittest.TestCase):
    def arguments(self, engine):
        return SimpleNamespace(engine=engine, model=Path("/models/x"), tokens=2)

    def group(self):
        return [
            {"case": "batch4", "repeat": 0, "slot": slot, "tokens": [slot + 1]} for slot in range(4)
        ]

    def response(self, engine):
        if engine == "native":
            events = [
                {"Token": {"token": 11}},
                {"Token": {"token": 12}},
                {
                    "Finished": {
                        "reason": "Length",
                        "measurement": {
                            "successful": True,
                            "ttft_us": 1,
                            "e2e_us": 2,
                            "max_tpot_us": 1,
                            "output_tokens": 2,
                        },
                    }
                },
            ]
        else:
            events = [{"choices": [{"token_ids": [11, 12], "finish_reason": "length"}]}]
        lines = [b"data: " + json.dumps(event).encode() + b"\n" for event in events]
        response = mock.MagicMock()
        response.__enter__.return_value = lines
        response.__exit__.return_value = False
        return response

    def token(self, payload, engine):
        return (
            payload["input"]["Sequence"]["tokens"][0]
            if engine == "native"
            else payload["prompt"][0]
        )

    def test_group_prepares_all_payloads_before_http_and_records_actual_send_times(self):
        for engine in ["native", "vllm"]:
            with self.subTest(engine=engine):
                self.check_group(engine)

    def check_group(self, engine):
        prepared = {}
        sends = {}
        fast_ready = threading.Event()
        lock = threading.Lock()
        serialize = json.dumps
        response = self.response(engine)

        def prepare(payload):
            token = self.token(payload, engine)
            if token == 4:
                self.assertTrue(fast_ready.wait(timeout=2))
                with lock:
                    self.assertFalse(sends, "early requests bypassed preparation barrier")
            data = serialize(payload)
            with lock:
                prepared[token] = time.perf_counter_ns()
                if len(prepared) == 3:
                    fast_ready.set()
            return data

        def send(request, timeout):
            token = self.token(json.loads(request.data), engine)
            with lock:
                self.assertEqual(len(prepared), 4)
                sends[token] = time.perf_counter_ns()
            self.assertEqual(timeout, 600)
            self.assertEqual(request.get_header("Content-type"), "application/json")
            return response

        with (
            mock.patch.object(module.json, "dumps", side_effect=prepare),
            mock.patch.object(module.urllib.request, "urlopen", side_effect=send),
        ):
            timing = module.measure_group(
                self.arguments(engine), "http://test", [], iter(range(1, 5)), self.group()
            )
        release = timing["group_start_monotonic_ns"]
        self.assertGreaterEqual(release, max(prepared.values()))
        self.assertEqual(len(sends), 4)
        self.assertEqual(timing["start_synchronization"], "barrier_after_payload_serialization")
        starts = []
        ends = []
        for row in timing["results"]:
            token = row["slot"] + 1
            start = row["request_start_monotonic_ns"]
            end = row["request_end_monotonic_ns"]
            self.assertLessEqual(release, start)
            self.assertLessEqual(start, sends[token])
            self.assertLessEqual(sends[token], end)
            self.assertLessEqual(row["request_start_unix_ns"], row["request_end_unix_ns"])
            self.assertEqual(row["token_ids"], [11, 12])
            self.assertEqual(row["request_wall_seconds"], (end - start) / 1e9)
            starts.append(start)
            ends.append(end)
        self.assertEqual(timing["request_start_skew_seconds"], (max(starts) - min(starts)) / 1e9)
        self.assertEqual(timing["group_makespan_seconds"], (max(ends) - release) / 1e9)
        self.assertEqual(timing["wall_seconds"], timing["group_makespan_seconds"])

    def test_payload_error_releases_waiting_workers_and_preserves_original_error(self):
        serialize = json.dumps

        def prepare(payload):
            if payload["prompt"] == [4]:
                raise ValueError("payload preparation failed")
            return serialize(payload)

        with (
            mock.patch.object(module.json, "dumps", side_effect=prepare),
            mock.patch.object(module.urllib.request, "urlopen") as send,
            self.assertRaisesRegex(ValueError, "payload preparation failed"),
        ):
            module.measure_group(
                self.arguments("vllm"), "http://test", [], iter(range(1, 5)), self.group()
            )
        send.assert_not_called()

    def test_direct_single_request_call_remains_supported(self):
        for engine in ["native", "vllm"]:
            with (
                self.subTest(engine=engine),
                mock.patch.object(
                    module.urllib.request, "urlopen", return_value=self.response(engine)
                ) as send,
            ):
                result = module.stream_request(
                    self.arguments(engine),
                    "http://test",
                    Path("/models/x"),
                    [],
                    iter([7]),
                    self.group()[0],
                )
                self.assertEqual(result["token_ids"], [11, 12])
                self.assertEqual(result["request_id"], 7 if engine == "native" else None)
                request = send.call_args.args[0]
                self.assertTrue(
                    request.full_url.endswith(
                        "/native/v1/stream" if engine == "native" else "/v1/completions"
                    )
                )


if __name__ == "__main__":
    unittest.main()
