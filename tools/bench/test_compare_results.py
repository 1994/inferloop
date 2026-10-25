"""Reject incomparable experiments and performance/numerical regressions."""

import copy
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "compare_results", Path(__file__).with_name("compare-results.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def identity(engine="native", **overrides):
    block = {
        "run_id": "run-1",
        "profile_id": "profile-1",
        "engine": engine,
        "engine_version": "0.1.0",
        "source": {"revision": "deadbeef", "dirty": False},
        "release_profile": {
            "source": "fixture",
            "effective": {
                "opt-level": 3,
                "lto": "thin",
                "codegen-units": 1,
                "debug": False,
                "debug-assertions": False,
                "overflow-checks": False,
            },
            "explicit": {"lto": "thin", "codegen-units": 1},
        },
        "build": {"path": "/bin/infer", "sha256": "abc", "release_like": True},
        "model": {"path": "/models/x", "files": {"config.json": "aaa"}, "shards": {"a": 1}},
        "hardware": {"devices": [{"uuid": "GPU-1", "name": "RTX 5090", "driver_version": "1"}]},
        "limits": {"source": "native runtime inspection", "effective_max_model_len": 8192},
        "cache": {"requested": True, "effective": True},
        "verified": True,
    }
    for key, value in overrides.items():
        if value is None:
            block.pop(key, None)
        else:
            block[key] = value
    return block


def report(case="hot_long", concurrency=1, tokens=2, repeats=3, engine="native"):
    """One report with the identity, matrix and per-request evidence the gate now requires."""
    return {
        "completed": True,
        "model": "fixture",
        "inputs_sha256": "same-input",
        "max_new_tokens": tokens,
        "temperature": 0,
        "mtp_depth": 2,
        "prefix_cache_enabled": True,
        "eos_tokens": [9],
        "hot_prefix_tokens_reused": 20,
        "gpu_memory_utilization": 0.88,
        "matrix": {case: {"concurrency": concurrency, "repeats": repeats}},
        "identity": identity(engine),
        # Both sides of a pair must have matched the same declared profile.
        "checklist": {"path": "profile.json", "sha256": "checklist-1", "compliant": True},
        "telemetry": {
            "log": "run.telemetry.jsonl",
            "sha256": "telemetry-1",
            "samples": 40,
            "active_window": {"samples": 30, "utilization_gpu_mean": 42.0},
            "active_process": {"samples": 30, "cpu_percent_one_core_mean": 120.0},
            "measured_span": {"samples": 40, "utilization_gpu_mean": 30.0},
        },
        "trials": [
            {
                "case": case,
                "repeat": repeat,
                "warmup": False,
                "concurrency": concurrency,
                "wall_seconds": 1.0,
                "prefix_tokens_reused": 20 if case == "hot_long" else 0,
                "gpu": {
                    "utilization_gpu_mean": 42.0,
                    "utilization_gpu_max": 88.0,
                    "utilization_memory_mean": 30.0,
                    "sm_clock_mean_mhz": 2400.0,
                },
                "server_process": {"cpu_percent_one_core_mean": 120.0},
                "results": [
                    {
                        "slot": slot,
                        "input_tokens": 20,
                        "output_tokens_excluding_eos": tokens,
                        "token_ids": list(range(1, tokens + 1)),
                        "ttft_seconds": 0.1,
                        "tpot_seconds": 0.9 if tokens > 1 else None,
                        "finish_reason": "length",
                        "truncated": False,
                        "server_measurement": {
                            "ttft_seconds": 0.05,
                            "e2e_seconds": 0.5,
                            "output_tokens": tokens,
                        },
                    }
                    for slot in range(concurrency)
                ],
            }
            for repeat in range(repeats)
        ],
    }


class GateTests(unittest.TestCase):
    def test_equal_and_regressed_latency(self):
        old = report()
        new = copy.deepcopy(old)
        self.assertTrue(module.compare(old, new)["passed"])
        for trial in new["trials"]:
            for result in trial["results"]:
                result["tpot_seconds"] *= 1.2
        result = module.compare(old, new)
        self.assertFalse(result["passed"])
        self.assertFalse(result["performance"]["passed"])

    def test_shorter_generation_cannot_pass_as_faster(self):
        old, new = report(), report()
        new["trials"][0]["results"][0]["output_tokens_excluding_eos"] = 1
        new["trials"][0]["results"][0]["token_ids"] = [1]
        new["trials"][0]["results"][0]["server_measurement"]["output_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "request work differs"):
            module.compare(old, new)

    def test_actual_cache_reuse_and_input_alignment_required(self):
        for field, value in [
            ("inputs_sha256", "other"),
            ("mtp_depth", 0),
            ("completed", False),
            ("hot_prefix_tokens_reused", 0),
            ("vision_enabled", True),
        ]:
            with self.subTest(field=field):
                old, new = report(), report()
                new[field] = value
                with self.assertRaises(ValueError):
                    module.compare(old, new)

    def test_a_report_without_an_effective_profile_is_legacy_not_invalid(self):
        # The plan keeps old profiles' results, so an absent profile classifies as legacy; a profile
        # that is present but wrong stays invalid, which the next two cases check.
        legacy = report() | {
            "identity": identity(release_profile={"lto": "thin", "codegen-units": 1})
        }
        module.require_release_identity(legacy)
        self.assertEqual(module.release_profile_state(legacy["identity"]), "legacy")

    def test_a_debug_assertion_build_is_rejected(self):
        profile = identity()["release_profile"]
        profile["effective"]["debug-assertions"] = True
        with self.assertRaises(ValueError) as raised:
            module.release_profile_state(identity(release_profile=profile))
        self.assertIn("debug assertions", str(raised.exception))

    def test_an_unoptimized_build_is_rejected(self):
        profile = identity()["release_profile"]
        profile["effective"]["opt-level"] = 0
        with self.assertRaises(ValueError) as raised:
            module.release_profile_state(identity(release_profile=profile))
        self.assertIn("without optimization", str(raised.exception))

    def test_a_legacy_pair_needs_the_historical_flag(self):
        legacy = report() | {"identity": identity(release_profile={"lto": "thin"})}
        with self.assertRaises(ValueError) as raised:
            module.compare(legacy, legacy)
        self.assertIn("predates the effective-profile rule", str(raised.exception))
        # With the flag the same frozen pair is a historical comparison, which the plan keeps.
        result = module.compare(legacy, legacy, allow_legacy_baseline=True)
        self.assertIn("metrics", result)

    def test_a_legacy_candidate_is_refused_while_it_is_not_a_historical_comparison(self):
        fresh = report()
        legacy = report() | {"identity": identity(release_profile={"lto": "thin"})}
        with self.assertRaises(ValueError) as raised:
            module.compare(fresh, legacy)
        self.assertIn("candidate report predates", str(raised.exception))

    def test_numerical_gate(self):
        old, new = report(), report()
        new["trials"][0]["results"][0]["token_ids"] = [1, 3]
        result = module.compare(old, new, identical=True)
        self.assertFalse(result["passed"])
        self.assertFalse(result["numeric"]["identical"])
        self.assertEqual(result["token_mismatches"], 1)
        # Performance and numeric results stay separate conclusions.
        self.assertTrue(result["performance"]["passed"])

    def test_missing_samples_and_nan_rejected(self):
        old, new = report(), report()
        new["trials"].pop()
        with self.assertRaises(ValueError):
            module.compare(old, new)
        new = report()
        new["trials"][0]["wall_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            module.compare(old, new)

    def test_a_single_slot_group_is_not_the_declared_case(self):
        old = report(case="batch4", concurrency=4)
        new = copy.deepcopy(old)
        new["trials"][0]["results"] = new["trials"][0]["results"][:1]
        with self.assertRaisesRegex(ValueError, "expected 4"):
            module.compare(old, new)

    def test_a_hot_repeat_without_reuse_is_rejected(self):
        old, new = report(), report()
        new["trials"][1]["prefix_tokens_reused"] = 0
        with self.assertRaisesRegex(ValueError, "without observed reuse"):
            module.compare(old, new)

    def test_a_trial_without_hardware_telemetry_is_rejected(self):
        # A latency difference with no device evidence cannot be attributed to the engine.
        old, new = report(), report()
        new["trials"][0]["gpu"] = None
        with self.assertRaisesRegex(ValueError, "hardware telemetry"):
            module.compare(old, new)
        new = report()
        new["trials"][0]["gpu"].pop("utilization_gpu_mean")
        with self.assertRaisesRegex(ValueError, "hardware telemetry"):
            module.compare(old, new)

    def test_the_result_compares_device_utilisation(self):
        old, new = report(), report()
        for trial in new["trials"]:
            trial["gpu"]["utilization_gpu_mean"] = 70.0
        result = module.compare(old, new)
        row = next(entry for entry in result["hardware"]["per_case"] if entry["case"] == "hot_long")
        self.assertEqual(
            result["hardware"]["active_window"]["baseline"]["utilization_gpu_mean"], 42.0
        )
        self.assertEqual(
            result["hardware"]["active_process"]["candidate"]["cpu_percent_one_core_mean"], 120.0
        )
        self.assertEqual(row["baseline_utilization_gpu_mean"], 42.0)
        self.assertEqual(row["candidate_utilization_gpu_mean"], 70.0)
        self.assertIn("baseline_sm_clock_mean_mhz", row)

    def test_a_truncated_request_is_rejected(self):
        old, new = report(), report()
        new["trials"][0]["results"][0]["truncated"] = True
        with self.assertRaisesRegex(ValueError, "truncated"):
            module.compare(old, new)
        new = report()
        new["trials"][0]["results"][0]["finish_reason"] = None
        with self.assertRaises(ValueError):
            module.compare(old, new)

    def test_a_server_that_reports_more_tokens_than_it_streamed_is_rejected(self):
        old, new = report(), report()
        new["trials"][0]["results"][0]["server_measurement"]["output_tokens"] = 5
        with self.assertRaisesRegex(ValueError, "were streamed"):
            module.compare(old, new)

    def test_a_matrix_row_that_was_never_measured_is_rejected(self):
        old, new = report(), report()
        new["matrix"]["hot_long"]["repeats"] = 4
        with self.assertRaisesRegex(ValueError, "was not measured"):
            module.compare(old, new)

    def test_a_single_visible_token_leaves_tpot_unavailable(self):
        old = report(tokens=1)
        new = copy.deepcopy(old)
        result = module.compare(old, new)
        self.assertTrue(result["passed"])
        tpot = next(m for m in result["metrics"] if m["metric"] == "tpot_seconds")
        self.assertFalse(tpot["available"])
        # The other metrics are still judged.
        self.assertTrue(any(m["available"] for m in result["metrics"]))

    def test_an_unavailable_metric_does_not_mask_a_regression(self):
        old = report(tokens=1)
        new = copy.deepcopy(old)
        for trial in new["trials"]:
            trial["wall_seconds"] = 2.0
        self.assertFalse(module.compare(old, new)["passed"])


class OrientationTest(unittest.TestCase):
    """The target is native/vLLM <= 1.10x, so the direction must not be guessable."""

    @staticmethod
    def pair(native_scale=1.0, vllm_scale=1.0):
        native = report()
        vllm = report()
        vllm["identity"] = identity(
            "vllm",
            limits={"source": "command line", "max_model_len": 8192},
            build={"path": "/venv/bin/python", "sha256": "x"},
        )
        for trial in native["trials"]:
            trial["wall_seconds"] *= native_scale
        for trial in vllm["trials"]:
            trial["wall_seconds"] *= vllm_scale
        return native, vllm

    def test_a_slower_engine_under_test_fails_the_target(self):
        # native is 40% slower than vLLM: it must fail, not pass by inversion.
        native, vllm = self.pair(native_scale=1.4)
        result = module.compare(native, vllm)
        self.assertEqual(result["ratio_orientation"], "native/vllm")
        self.assertAlmostEqual(result["metrics"][0]["ratio"], 1.4)
        self.assertFalse(result["passed"])
        # Writing the arguments the other way round must not change the verdict.
        swapped = module.compare(vllm, native)
        self.assertEqual(swapped["ratio_orientation"], "native/vllm")
        self.assertAlmostEqual(swapped["metrics"][0]["ratio"], 1.4)
        self.assertFalse(swapped["passed"])

    def test_a_faster_engine_under_test_passes(self):
        native, vllm = self.pair(native_scale=0.8)
        self.assertTrue(module.compare(native, vllm)["passed"])

    def test_an_unnamed_reference_is_refused(self):
        native, sglang = self.pair()
        sglang["identity"]["engine"] = "sglang"
        with self.assertRaisesRegex(ValueError, "neither report is the reference"):
            module.compare(native, sglang)

    def test_a_same_engine_pair_is_an_a_b(self):
        old, new = report(), report()
        for trial in new["trials"]:
            trial["wall_seconds"] *= 1.3
        result = module.compare(old, new)
        self.assertEqual(result["ratio_orientation"], "native/baseline")
        self.assertAlmostEqual(result["metrics"][0]["ratio"], 1.3)


def native_resources():
    # Startup inspection: pool_bytes is the backend slot pool's block-vocabulary
    # capacity, not a measurement of physical device allocation.
    return {
        "source": "native runtime inspection",
        "readback": {
            "scheduler": {"prefill_chunk_tokens": 1024, "forward_chunk_tokens": 64},
            "execution_profile": {
                "prefill_width": 1024,
                "pooled_prefill_width": 1024,
                "batch_width": 3,
                "draft_depth": 2,
                "arena_budget_bytes": 1054928896,
                "attention_profile": "bf16",
                "precise_prefix_positions": 2048,
            },
            "serving_plan": {
                "source": "native server log",
                "width": 4,
                "capacity": 8192,
                "kv_tokens": 20512,
                "prompt_width": 1024,
                "quoted_bytes": 3384849456,
                "envelope_bytes": 4493738323,
                "allocated_bytes": 4184529176,
            },
            "kv_cache": {
                "block_size": 32,
                "total_blocks": 8192,
                "available_blocks": 1024,
                "pool_bytes": 1073741824,
                "bytes_per_block": 1048576,
            },
        },
    }


class ResourceAlignmentTest(unittest.TestCase):
    @staticmethod
    def pair():
        old, new = report(), report()
        old["config_readback"] = native_resources()
        new["config_readback"] = native_resources()
        new["config_readback"]["readback"]["execution_profile"]["precise_prefix_positions"] = 0
        return old, new

    def test_fixed_resource_pair_rejects_observed_capture_replan(self):
        old, new = self.pair()
        # PR47 pair00: identical declared limits, but 4 versus 7 captured slots.
        kv = new["config_readback"]["readback"]["kv_cache"]
        kv.update(pool_bytes=1879048192, available_blocks=1792)
        with self.assertRaisesRegex(ValueError, "native resource.*pool_bytes"):
            module.compare(old, new, require_identical_resources=True)

    def test_a_numeric_candidate_can_keep_the_same_resource_plan(self):
        old, new = self.pair()
        self.assertTrue(module.compare(old, new, require_identical_resources=True)["passed"])

    def test_each_execution_geometry_and_capacity_difference_is_rejected(self):
        for section, fields in (
            (
                "execution_profile",
                (
                    "prefill_width",
                    "pooled_prefill_width",
                    "batch_width",
                    "draft_depth",
                    "arena_budget_bytes",
                ),
            ),
            ("kv_cache", ("block_size", "total_blocks", "pool_bytes", "bytes_per_block")),
            (
                "serving_plan",
                (
                    "width",
                    "capacity",
                    "kv_tokens",
                    "prompt_width",
                    "quoted_bytes",
                    "envelope_bytes",
                ),
            ),
        ):
            for field in fields:
                with self.subTest(section=section, field=field):
                    old, new = self.pair()
                    new["config_readback"]["readback"][section][field] *= 2
                    with self.assertRaisesRegex(ValueError, "native resource.*" + field):
                        module.compare(old, new, require_identical_resources=True)

    def test_a_scheduler_change_is_a_different_fixed_resource_experiment(self):
        old, new = self.pair()
        new["config_readback"]["readback"]["scheduler"]["forward_chunk_tokens"] = 128
        with self.assertRaisesRegex(ValueError, "native resource.*scheduler"):
            module.compare(old, new, require_identical_resources=True)

    def test_strict_resources_require_complete_valid_readback(self):
        for mutate in (
            lambda report: report.pop("config_readback"),
            lambda report: report.update(config_readback=["invalid"]),
            lambda report: report["config_readback"].update(readback=None),
            lambda report: report["config_readback"]["readback"].pop("scheduler"),
            lambda report: report["config_readback"]["readback"]["execution_profile"].pop(
                "batch_width"
            ),
            lambda report: report["config_readback"]["readback"]["kv_cache"].update(
                pool_bytes=None
            ),
            lambda report: report["config_readback"]["readback"]["kv_cache"].update(
                pool_bytes=True
            ),
        ):
            with self.subTest(mutate=mutate):
                old, new = self.pair()
                mutate(new)
                with self.assertRaisesRegex(ValueError, "native resource"):
                    module.compare(old, new, require_identical_resources=True)

    def test_resource_changing_experiments_and_historical_pairs_remain_available(self):
        old, new = self.pair()
        new["config_readback"]["readback"]["kv_cache"]["pool_bytes"] *= 2
        self.assertTrue(module.compare(old, new)["passed"])
        self.assertTrue(module.compare(report(), report())["passed"])

    def test_dynamic_cache_activity_is_not_a_capacity_change(self):
        old, new = self.pair()
        new["config_readback"]["readback"]["kv_cache"].update(
            available_blocks=512, active_blocks=512, reused_tokens=256
        )
        self.assertTrue(module.compare(old, new, require_identical_resources=True)["passed"])

    def test_physical_measurement_noise_does_not_change_the_resource_contract(self):
        old, new = self.pair()
        new["config_readback"]["readback"]["serving_plan"]["allocated_bytes"] += 1024
        self.assertTrue(module.compare(old, new, require_identical_resources=True)["passed"])

    def test_cross_engine_comparison_keeps_distinct_resource_implementations(self):
        native, reference = OrientationTest.pair()
        native["config_readback"] = native_resources()
        reference["config_readback"] = {"source": "command line", "readback": None}
        self.assertTrue(module.compare(native, reference)["passed"])


class IdentityAlignmentTest(unittest.TestCase):
    def test_a_missing_identity_block_is_rejected(self):
        old, new = report(), report()
        del new["identity"]
        with self.assertRaisesRegex(ValueError, "identity block"):
            module.compare(old, new)

    def test_an_unverified_or_debug_build_is_rejected(self):
        for mutate in (
            lambda block: block.update(verified=False),
            lambda block: block["build"].update(release_like=False),
            lambda block: block["build"].pop("release_like"),
            lambda block: block.update(source={"revision": None, "dirty": True}),
            lambda block: block.update(hardware={"devices": []}),
            lambda block: block.update(model={"path": "/models/x", "files": {}}),
        ):
            with self.subTest(mutate=mutate):
                old, new = report(), report()
                mutate(new["identity"])
                with self.assertRaises(ValueError):
                    module.compare(old, new)

    def test_a_different_model_artifact_is_rejected(self):
        old, new = report(), report()
        new["identity"]["model"]["files"]["config.json"] = "bbb"
        with self.assertRaisesRegex(ValueError, "model identity"):
            module.compare(old, new)

    def test_a_different_gpu_is_rejected(self):
        old, new = report(), report()
        new["identity"]["hardware"]["devices"][0]["uuid"] = "GPU-2"
        with self.assertRaisesRegex(ValueError, "hardware identity"):
            module.compare(old, new)

    def test_a_different_resource_constraint_is_rejected(self):
        old, new = report(), report()
        new["gpu_memory_utilization"] = 0.5
        with self.assertRaisesRegex(ValueError, "gpu_memory_utilization"):
            module.compare(old, new)

    def test_same_engine_requires_identical_serving_limits(self):
        old, new = report(), report()
        new["identity"]["limits"] = {
            "source": "native runtime inspection",
            "effective_max_model_len": 4096,
        }
        with self.assertRaisesRegex(ValueError, "serving limits"):
            module.compare(old, new)

    def test_cross_engine_needs_a_profile_and_an_aligned_context(self):
        old = report()
        new = copy.deepcopy(old)
        new["identity"] = identity(
            "vllm",
            limits={"source": "command line", "max_model_len": 8192},
            build={"path": "/venv/bin/python", "sha256": "x"},
        )
        module.compare(old, new)
        # A missing shared profile makes the pairing unverifiable.
        new["identity"]["profile_id"] = None
        with self.assertRaisesRegex(ValueError, "profile id"):
            module.compare(old, new)
        # A different context ceiling is a different experiment.
        new["identity"] = identity(
            "vllm",
            limits={"source": "command line", "max_model_len": 4096},
            build={"path": "/venv/bin/python", "sha256": "x"},
        )
        with self.assertRaisesRegex(ValueError, "not aligned across engines"):
            module.compare(old, new)


if __name__ == "__main__":
    unittest.main()
