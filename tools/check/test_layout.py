"""Architecture gates prevent model logic from returning to orchestration."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from layout import check_cuda_hardware_boundary, check_ownership, production_dependencies


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


class CudaHardwareBoundaryTests(unittest.TestCase):
    def test_queries_and_platform_selection_belong_to_device_boundary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "crates/backend/cuda/src"
            (source / "device").mkdir(parents=True)
            (source / "device.rs").write_text("cuDeviceGetAttribute();\n")
            (source / "device/platform.rs").write_text('cfg!(target_os = "linux");\n')
            (source / "device/library_names.rs").write_text('"libnvjpeg.so.13"\n')
            other = source / "vision.rs"
            # Selection from facts already returned by the device is permitted here.
            other.write_text("is_sm120(&profile.architecture);\n")
            check_cuda_hardware_boundary(root)
            for statement in [
                "cuDeviceGetAttribute();",
                "cuDeviceComputeCapability();",
                "cuDriverGetVersion();",
                "cudaGetDeviceProperties();",
                "cudaDriverGetVersion();",
                "nvmlDeviceGetCount_v2();",
                "nvjpegGetHardwareDecoderInfo();",
                "nvjpegDecoderJpegSupported();",
                "nvjpegDecodeBatchedSupportedEx();",
                'cfg!(target_os = "windows");',
                "#[cfg(windows)]",
                '#[cfg(all(windows, feature = "cuda"))]',
                '#[cfg_attr(windows, link(name = "nvjpeg"))]',
                "std::env::consts::OS;",
                '"nvjpeg64_13.dll"',
                '"libnvjpeg.so.13"',
                '"libcuda.so.1"',
            ]:
                with self.subTest(statement=statement):
                    other.write_text(statement + "\n")
                    with self.assertRaises(SystemExit):
                        check_cuda_hardware_boundary(root)


if __name__ == "__main__":
    unittest.main()
