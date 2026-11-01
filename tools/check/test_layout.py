"""Architecture gates prevent model logic from returning to orchestration."""

import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

from layout import (
    CUDA_HOST,
    check_cuda_host_ratchet,
    check_host_architecture_names,
    check_ownership,
    check_workload_blindness,
    production_dependencies,
)


class OwnershipTests(unittest.TestCase):
    def test_allowed_dependencies(self):
        for member, deps in [
            ("crates/model/package", {"infer-model-recipes", "infer-spi"}),
            ("crates/model/compiler", {"infer-ir", "infer-kernel-api"}),
            ("crates/engine/runtime", {"infer-compiler", "infer-state"}),
            (
                "crates/engine/workloads",
                {"infer-core", "infer-ir", "infer-spi", "infer-models"},
            ),
        ]:
            check_ownership(member, deps)

    def test_model_and_device_logic_cannot_leak_into_core(self):
        for member, dependency in [
            ("crates/model/compiler", "infer-model-recipes"),
            ("crates/model/package", "infer-compiler"),
            ("crates/engine/runtime", "infer-models"),
            ("crates/engine/scheduler", "infer-kernel-api"),
            ("crates/engine/state", "infer-backend-cuda"),
            ("crates/foundation/spi", "infer-runtime"),
            ("crates/backend/kernel-api", "infer-backend-cuda"),
            # I1: the common algorithm layer plans over the shared contracts only.
            ("crates/engine/workloads", "infer-backend-cuda"),
            ("crates/engine/workloads", "infer-backend-metal"),
            ("crates/engine/workloads", "infer-kernel-api"),
            ("crates/engine/workloads", "infer-runtime"),
            ("crates/engine/runtime", "infer-backend-cuda"),
            ("crates/engine/runtime", "infer-backend-metal"),
        ]:
            with self.subTest(member=member), self.assertRaises(SystemExit):
                check_ownership(member, {dependency})

    def test_target_specific_dependencies_cannot_bypass_gate(self):
        dependencies = production_dependencies(
            {
                "dev-dependencies": {"infer-models": {}},
                "target": {"cfg(unix)": {"dependencies": {"infer-model-recipes": {}}}},
            }
        )
        self.assertEqual(dependencies, {"infer-model-recipes"})
        with self.assertRaises(SystemExit):
            check_ownership("crates/engine/runtime", dependencies)


class HostBoundaryTests(unittest.TestCase):
    BASELINE: ClassVar = {
        "Tensor<": 1,
        ".partition(": 0,
        ".generics(": 0,
        "cuda_core": 0,
        "CudaGraph": 0,
    }

    def tree(self, source):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        host = root / CUDA_HOST
        host.mkdir(parents=True)
        (root / "crates/engine/runtime/src").mkdir(parents=True)
        (host / "lib.rs").write_text(source)
        return root

    def test_the_baseline_holds(self):
        check_cuda_host_ratchet(self.tree("let t: Tensor<f32>;"), self.BASELINE)

    def test_new_host_recording_fails(self):
        with self.assertRaisesRegex(SystemExit, "exceed the baseline"):
            check_cuda_host_ratchet(self.tree("Tensor<f32> Tensor<f32>"), self.BASELINE)

    def test_a_gain_must_lower_the_baseline(self):
        with self.assertRaisesRegex(SystemExit, "lower CUDA_HOST_TOKENS"):
            check_cuda_host_ratchet(self.tree(""), self.BASELINE)

    def test_only_artifact_catalogs_name_an_architecture(self):
        root = self.tree('let arch = "sm_120";')
        with self.assertRaisesRegex(SystemExit, "ask contract::target::Arch"):
            check_host_architecture_names(root)
        catalog = self.tree("// sm_120 in a comment")
        directory = catalog / CUDA_HOST / "resident/attention_mma"
        directory.mkdir(parents=True)
        (directory / "catalog.rs").write_text('"sm_120" => &[Self::Sm120],')
        check_host_architecture_names(catalog)

    def test_workload_branches_fail_but_comments_pass(self):
        check_workload_blindness(self.tree("// WorkloadKind stays elsewhere"))
        with self.assertRaisesRegex(SystemExit, "must not branch"):
            check_workload_blindness(self.tree("match kind { WorkloadKind::Embed => {} }"))


if __name__ == "__main__":
    unittest.main()
