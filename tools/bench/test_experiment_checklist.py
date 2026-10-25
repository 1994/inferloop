"""A run must prove it matched the profile it claims, not just that it completed."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "experiment_checklist", Path(__file__).with_name("experiment-checklist.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

gate_spec = importlib.util.spec_from_file_location(
    "gate_fixture", Path(__file__).with_name("test_compare_results.py")
)
gate = importlib.util.module_from_spec(gate_spec)
gate_spec.loader.exec_module(gate)


def checklist():
    return {
        "schema": 1,
        "profile_id": "fixture-v1",
        "model": {"path": "fixture", "files": {"config.json": "aaa"}},
        "numeric": {
            "weights": "nvfp4 storage, re-quantization recorded per profile",
            "activation": "fp8 with per-channel scales, fp32 accumulation",
            "kv": "model default precision, recorded by the run",
        },
        "quality": {"gate": "own gate", "required": False},
        "resources": {
            "gpu_memory_utilization": 0.88,
            "max_model_len": 8192,
            "max_num_seqs": 16,
        },
        "cache": {"prefix_cache": True},
        "workload": {
            "inputs_sha256": "same-input",
            "max_new_tokens": 2,
            "temperature": 0,
            "mtp_depth": 2,
            "eos_tokens": [9],
            "matrix": {"hot_long": {"concurrency": 1, "repeats": 3}},
        },
        "engines": {"native": {}, "vllm": {"version": "0.31.0"}},
        "measurement": {"repeats": 3, "max_ratio": 1.1, "require_identical_tokens": False},
    }


def native_report():
    report = gate.report()
    report["matrix"] = {"hot_long": {"concurrency": 1, "repeats": 3}}
    report["identity"]["limits"] = {
        "source": "native runtime inspection",
        "effective_max_model_len": 8192,
        "effective_input_cap": 8192,
        "effective_output_cap": 8192,
    }
    return report


def reference_report():
    report = copy.deepcopy(native_report())
    report["identity"] = gate.identity(
        "vllm",
        limits={
            "source": "declared on the command line",
            "max_model_len": 8192,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 2048,
        },
        build={"path": "/venv/bin/python", "sha256": "x"},
        engine_version="0.31.0",
    )
    return report


class ValidateTest(unittest.TestCase):
    def test_a_missing_section_is_rejected_rather_than_defaulted(self):
        with tempfile.TemporaryDirectory() as directory:
            for section in module.REQUIRED_SECTIONS:
                document = checklist()
                del document[section]
                path = Path(directory) / "c.json"
                path.write_text(json.dumps(document))
                with self.subTest(section=section), self.assertRaises(ValueError):
                    module.load(path)

    def test_a_well_formed_checklist_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.json"
            path.write_text(json.dumps(checklist()))
            self.assertEqual(module.load(path)["profile_id"], "fixture-v1")


class CheckRunTest(unittest.TestCase):
    def test_a_matching_run_has_no_mismatches(self):
        self.assertEqual(module.check_run(checklist(), native_report()), [])
        self.assertEqual(module.check_run(checklist(), reference_report()), [])

    def test_each_declared_condition_is_compared(self):
        for mutate in (
            lambda r: r.update(model="elsewhere"),
            lambda r: r["identity"]["model"]["files"].update({"config.json": "bbb"}),
            lambda r: r.update(gpu_memory_utilization=0.5),
            lambda r: r.update(prefix_cache_enabled=False),
            lambda r: r.update(max_new_tokens=8),
            lambda r: r.update(temperature=1),
            lambda r: r.update(mtp_depth=0),
            lambda r: r.update(eos_tokens=[7]),
            lambda r: r.update(inputs_sha256="other"),
            lambda r: r.update(matrix={"hot_long": {"concurrency": 2, "repeats": 3}}),
            lambda r: r["identity"]["limits"].update({"effective_max_model_len": 4096}),
        ):
            with self.subTest(mutate=mutate):
                report = native_report()
                mutate(report)
                self.assertTrue(module.check_run(checklist(), report))

    def test_the_reference_side_is_checked_against_its_own_declared_limits(self):
        report = reference_report()
        report["identity"]["limits"]["max_num_seqs"] = 4
        self.assertTrue(module.check_run(checklist(), report))
        report = reference_report()
        report["identity"]["engine_version"] = "0.30.0"
        self.assertTrue(module.check_run(checklist(), report))

    def test_declared_vision_and_reference_token_cap_are_verified(self):
        profile = checklist()
        profile["resources"].update(vision=True, max_num_batched_tokens=2048)
        report = reference_report()
        report["vision_enabled"] = True
        self.assertEqual(module.check_run(profile, report), [])
        report["vision_enabled"] = False
        self.assertTrue(module.check_run(profile, report))
        report["vision_enabled"] = True
        report["identity"]["limits"]["max_num_batched_tokens"] = 1024
        self.assertTrue(module.check_run(profile, report))


class ComparisonTableTest(unittest.TestCase):
    def test_aligned_engines_produce_an_aligned_table(self):
        rows = module.comparison_table([native_report(), reference_report()])
        by_item = {row["item"]: row for row in rows}
        for item in ("model", "prefix cache", "mtp depth", "output tokens", "workload inputs"):
            self.assertTrue(by_item[item]["aligned"], item)
        # The context ceiling is read back on one side and declared on the other, and still has to
        # agree; the engine version legitimately differs.
        self.assertTrue(by_item["context ceiling"]["aligned"])
        self.assertFalse(by_item["engine version"]["aligned"])

    def test_a_different_context_ceiling_is_visible_in_the_table(self):
        report = reference_report()
        report["identity"]["limits"]["max_model_len"] = 4096
        rows = module.comparison_table([native_report(), report])
        ceiling = next(row for row in rows if row["item"] == "context ceiling")
        self.assertFalse(ceiling["aligned"])
        self.assertIn("native", ceiling["values"])
        self.assertIn("vllm", ceiling["values"])

    def test_two_reports_from_one_engine_are_rejected(self):
        with self.assertRaises(ValueError):
            module.comparison_table([native_report(), native_report()])


class GateChecklistTest(unittest.TestCase):
    def test_the_gate_requires_both_sides_to_share_a_checklist(self):
        old, new = native_report(), reference_report()
        del new["checklist"]
        with self.assertRaisesRegex(ValueError, "checklist"):
            gate.module.compare(old, new)

    def test_a_noncompliant_run_never_passes_the_gate(self):
        old, new = native_report(), reference_report()
        old["checklist"] = {"sha256": "c", "compliant": True}
        new["checklist"] = {"sha256": "c", "compliant": False}
        with self.assertRaisesRegex(ValueError, "did not match its experiment checklist"):
            gate.module.compare(old, new)

    def test_different_checklists_cannot_be_paired(self):
        old, new = native_report(), reference_report()
        old["checklist"] = {"sha256": "c1", "compliant": True}
        new["checklist"] = {"sha256": "c2", "compliant": True}
        with self.assertRaisesRegex(ValueError, "different checklists"):
            gate.module.compare(old, new)

    def test_a_shared_checklist_passes(self):
        old, new = native_report(), reference_report()
        old["checklist"] = {"sha256": "c1", "compliant": True}
        new["checklist"] = {"sha256": "c1", "compliant": True}
        self.assertTrue(gate.module.compare(old, new)["passed"])


class ProcessContractTest(unittest.TestCase):
    def contract(self):
        profile = checklist()
        profile["measurement"].update(
            fresh_process_pairs=5,
            paired_confidence=0.95,
            primary={
                "case": "hot_long",
                "metric": "wall_seconds",
                "minimum_improvement_percent": 2,
            },
            guards=[
                {
                    "case": "hot_long",
                    "metric": "tpot_seconds",
                    "max_regression_percent": 2,
                }
            ],
        )
        return profile

    def test_a_frozen_process_contract_is_accepted(self):
        module.load_document(self.contract())

    def test_process_count_and_matrix_membership_are_enforced(self):
        profile = self.contract()
        profile["measurement"]["fresh_process_pairs"] = 4
        with self.assertRaisesRegex(ValueError, "at least five"):
            module.load_document(profile)
        profile = self.contract()
        profile["measurement"]["primary"]["case"] = "c16"
        with self.assertRaisesRegex(ValueError, "absent from the workload matrix"):
            module.load_document(profile)

    def test_frozen_orders_must_cover_each_pair_once_per_side(self):
        for orders in (
            None,
            "AB/BA/AB/BA/AB",
            [["baseline", "candidate"]] * 4,
            [["baseline", "baseline"]] * 5,
            [["candidate", "unknown"]] * 5,
        ):
            profile = self.contract()
            profile["measurement"]["process_pair_orders"] = orders
            with (
                self.subTest(orders=orders),
                self.assertRaisesRegex(ValueError, "process_pair_orders"),
            ):
                module.load_document(profile)

    def test_frozen_orders_balance_which_side_runs_first(self):
        for count, baseline_first in ((5, 5), (5, 4), (6, 4)):
            profile = self.contract()
            profile["measurement"].update(
                fresh_process_pairs=count,
                process_pair_orders=[["baseline", "candidate"]] * baseline_first
                + [["candidate", "baseline"]] * (count - baseline_first),
            )
            with (
                self.subTest(count=count, baseline_first=baseline_first),
                self.assertRaisesRegex(ValueError, "balanced"),
            ):
                module.load_document(profile)
        for count, baseline_first in ((5, 3), (5, 2), (6, 3)):
            profile = self.contract()
            profile["measurement"].update(
                fresh_process_pairs=count,
                process_pair_orders=[["baseline", "candidate"]] * baseline_first
                + [["candidate", "baseline"]] * (count - baseline_first),
            )
            module.load_document(profile)

    def test_resource_alignment_switch_is_not_truthy_text(self):
        for value in (None, "false", 0, 1):
            profile = self.contract()
            profile["measurement"]["require_identical_resources"] = value
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "require_identical_resources must be a boolean"),
            ):
                module.load_document(profile)
        for value in (False, True):
            profile = self.contract()
            profile["measurement"]["require_identical_resources"] = value
            module.load_document(profile)

    def test_orders_without_a_sample_count_are_rejected(self):
        profile = checklist()
        profile["measurement"]["process_pair_orders"] = [["baseline", "candidate"]] * 5
        with self.assertRaisesRegex(ValueError, "requires fresh_process_pairs"):
            module.load_document(profile)

    def test_invalid_guard_threshold_is_rejected(self):
        profile = self.contract()
        del profile["measurement"]["guards"][0]["max_regression_percent"]
        with self.assertRaisesRegex(ValueError, "non-negative"):
            module.load_document(profile)


class NumericContractTest(unittest.TestCase):
    """A numeric contract that is a dtype name answers nothing."""

    def test_a_bare_dtype_name_is_rejected(self):
        for field, value in (("weights", "nvfp4"), ("activation", "FP8"), ("kv", "bf16")):
            profile = checklist()
            profile["numeric"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError) as raised:
                module.load_document(profile)
            self.assertIn("only the dtype name", str(raised.exception))

    def test_a_described_rule_is_accepted(self):
        profile = checklist()
        profile["numeric"]["activation"] = "fp8 with fp32 accumulation"
        module.load_document(profile)

    def test_an_empty_field_is_rejected(self):
        profile = checklist()
        profile["numeric"]["kv"] = "  "
        with self.assertRaises(ValueError) as raised:
            module.load_document(profile)
        self.assertIn("missing kv", str(raised.exception))


class SamplingSeedTest(unittest.TestCase):
    """Sampling without a fixed seed cannot be aligned across engines."""

    def test_sampling_needs_a_seed(self):
        profile = checklist()
        profile["workload"]["temperature"] = 0.7
        with self.assertRaises(ValueError) as raised:
            module.load_document(profile)
        self.assertIn("without a seed", str(raised.exception))

    def test_sampling_with_a_seed_is_accepted(self):
        profile = checklist()
        profile["workload"]["temperature"] = 0.7
        profile["workload"]["seed"] = 20261010
        module.load_document(profile)

    def test_greedy_needs_no_seed(self):
        profile = checklist()
        profile["workload"]["temperature"] = 0
        module.load_document(profile)


class MtpCapabilityTest(unittest.TestCase):
    """A model without an MTP head must not be measured through a plain-decode fallback."""

    def model_dir(self, config):
        root = Path(tempfile.mkdtemp())
        (root / "config.json").write_text(json.dumps(config))
        return root

    def test_a_declared_mtp_head_allows_a_depth(self):
        profile = checklist()
        profile["model"]["path"] = str(
            self.model_dir({"text_config": {"mtp_num_hidden_layers": 1}})
        )
        module.load_document(profile)

    def test_a_positive_depth_without_an_mtp_head_is_rejected(self):
        profile = checklist()
        profile["model"]["path"] = str(self.model_dir({"text_config": {"hidden_size": 8}}))
        with self.assertRaises(ValueError) as raised:
            module.load_document(profile)
        self.assertIn("plain-decode fallback", str(raised.exception))

    def test_a_plain_decode_profile_needs_no_mtp_head(self):
        profile = checklist()
        profile["workload"]["mtp_depth"] = 0
        profile["model"]["path"] = str(self.model_dir({"text_config": {"hidden_size": 8}}))
        module.load_document(profile)

    def test_an_unreadable_configuration_is_not_judged(self):
        profile = checklist()
        profile["model"]["path"] = str(Path(tempfile.mkdtemp()) / "missing")
        module.load_document(profile)

    def test_the_checked_in_profiles_are_consistent(self):
        for path in sorted(Path("benchmarks/profiles").glob("*.json")):
            module.load_document(json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
