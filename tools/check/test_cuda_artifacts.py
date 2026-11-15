"""Regression tests for integrity failures and portable artifact verification."""

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "cuda"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("cuda_artifacts", TOOLS / "kernels.py")
kernels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernels)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.entries = {e["name"]: e for e in kernels.catalog()["kernels"]}

    def test_all_frozen_ptx_and_sources_verify_without_cuda(self):
        # On macOS the portable hash/target gate still checks the AOT bundles.
        with patch.object(kernels.sys, "platform", "darwin"):
            for entry in self.entries.values():
                self.assertFalse(kernels.verify(entry)["regenerated"])

    def test_shared_symbol_rename_is_the_only_allowed_oxide_difference(self):
        entry = self.entries["oxide-attention"]
        original = (kernels.ROOT / entry["name"] / entry["ptx"]).read_text()
        renamed = kernels.SHARED.sub(
            lambda m: "__shared_mem_0123456789abcdef_" + m.group(2), original
        )
        with tempfile.TemporaryDirectory() as tmp:
            generated = Path(tmp) / "kernel.ptx"
            generated.write_text(renamed)
            self.assertTrue(kernels.verify(entry, generated)["regenerated"])
            generated.write_text(renamed.replace(".target sm_120", ".target sm_90"))
            with self.assertRaisesRegex(ValueError, "differs beyond"):
                kernels.verify(entry, generated)

    def test_changed_shared_allocation_rejected(self):
        entry = self.entries["oxide-attention"]
        text = (kernels.ROOT / entry["name"] / entry["ptx"]).read_text()
        with self.assertRaisesRegex(ValueError, "allocation"):
            kernels.normalize(text.replace("[16]", "[32]"), entry)

    def test_nvcc_output_must_be_byte_identical(self):
        entry = self.entries["vector-attention"]
        with tempfile.TemporaryDirectory() as tmp:
            generated = Path(tmp) / "kernel.ptx"
            generated.write_bytes(
                (kernels.ROOT / entry["name"] / entry["ptx"]).read_bytes() + b"\n"
            )
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                kernels.verify(entry, generated)

    def test_unregistered_ptx_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "new").mkdir()
            (root / "new/kernel.ptx").write_text("unexpected")
            with (
                patch.object(kernels, "ROOT", root),
                self.assertRaisesRegex(ValueError, "directories differ"),
            ):
                kernels.catalog()

    def test_missing_registration_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "kernels.json"
            data = json.loads(kernels.CATALOG.read_text())
            data["kernels"].pop()
            catalog.write_text(json.dumps(data))
            with (
                patch.object(kernels, "CATALOG", catalog),
                self.assertRaisesRegex(ValueError, "directories differ"),
            ):
                kernels.catalog()

    def test_missing_provenance_cannot_bypass_source_and_ptx_identity(self):
        entry = self.entries["shared-attention"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / entry["name"]
            shutil.copytree(kernels.ROOT / entry["name"], directory)
            (directory / "provenance.json").unlink()
            with (
                patch.object(kernels, "ROOT", root),
                self.assertRaisesRegex(ValueError, "Missing source/PTX identity"),
            ):
                kernels.verify(entry)

    def test_manifest_must_bind_bridge_source_and_ptx(self):
        entry = self.entries["fa4-attention-f16"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / entry["name"]
            shutil.copytree(kernels.ROOT / entry["name"], directory)
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            del manifest["entries"][0]["files"]["bridge.cu"]
            path.write_text(json.dumps(manifest))
            with (
                patch.object(kernels, "ROOT", root),
                self.assertRaisesRegex(ValueError, "Missing source/PTX identity"),
            ):
                kernels.verify(entry)

    def test_hash_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "source.cu"
            file.write_text("original")
            expected = kernels.sha(file)
            file.write_text("modified")
            with self.assertRaisesRegex(ValueError, "Identity mismatch"):
                kernels.check_hash(file, expected)

    def test_failed_rebuild_keeps_failure_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "build"
            args = type("Args", (), {"output": out, "oxide_backend": None})()
            with self.assertRaisesRegex(ValueError, "oxide-backend"):
                kernels.rebuild(self.entries["oxide-attention"], args, "13.4.92")
            result = json.loads((out / "result.json").read_text())
            self.assertFalse(result["completed"])
            self.assertFalse(result["GPU_execution"])
            self.assertIn("oxide-backend", result["error"])
            with self.assertRaises(FileExistsError):
                kernels.rebuild(self.entries["oxide-attention"], args, "13.4.92")


if __name__ == "__main__":
    unittest.main()
