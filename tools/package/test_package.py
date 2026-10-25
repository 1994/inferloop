"""Test the real Rust target contract and fail-closed cross-platform archives."""

import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import package


def elf(arch="x86_64"):
    data = bytearray(32)
    data[:6] = b"\x7fELF\x02\x01"
    data[18:20] = (62 if arch == "x86_64" else 183).to_bytes(2, "little")
    return data


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package.planner()

    def test_rust_build_script_unit_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = str(Path(temporary) / "contract-tests")
            package.run(["rustc", "--edition", "2024", "--test", "build.rs", "-o", binary])
            package.run([binary])

    def test_target_not_host_selects_backend(self):
        arm = package.resolve("linux-cuda", "aarch64-unknown-linux-gnu.2.34")
        mac = package.resolve("macos-metal", "aarch64-apple-darwin")
        self.assertEqual(arm["arch"], "aarch64")
        self.assertEqual(arm["zig_target"], "aarch64-unknown-linux-gnu.2.34")
        self.assertEqual(mac["backend"], "metal")
        self.assertEqual(mac["features"], [])

    def test_native_entry_uses_the_plan_and_discovers_host_headers(self):
        plan = package.resolve("auto", "x86_64-unknown-linux-gnu")
        command = package.native_cargo(plan)
        # The native build takes the plan's features but no cross target, so its artifact stays at
        # the conventional path.
        self.assertEqual(command[:3], ["cargo", "build", "--locked"])
        self.assertNotIn("--target", command)
        self.assertIn("cuda", command)
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "stddef.h").write_text("")
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(package, "output", return_value=temporary),
            ):
                env = package.native_env(plan)
        self.assertIn(f"-isystem {temporary}", env["BINDGEN_EXTRA_CLANG_ARGS"])
        # The same production feature contract as packaging applies to a native build.
        self.assertEqual(env["INFER_PACKAGE_BUILD"], "1")

    def test_native_env_does_not_probe_gcc_for_metal(self):
        plan = package.resolve("auto", "aarch64-apple-darwin")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(package, "output", side_effect=AssertionError("no gcc probe")),
        ):
            env = package.native_env(plan)
        self.assertNotIn("BINDGEN_EXTRA_CLANG_ARGS", env)

    def test_production_features_and_zig_abi_are_explicit(self):
        plan = package.resolve("auto", "x86_64-unknown-linux-gnu")
        command = package.cargo("zigbuild", plan)
        self.assertEqual(command[:2], ["cargo", "zigbuild"])
        self.assertIn("x86_64-unknown-linux-gnu.2.28", command)
        self.assertIn("--no-default-features", command)
        self.assertEqual(command[-2:], ["--features", "cuda"])
        self.assertNotIn("test-backends", command)

    def test_macos_packages_use_apple_linker_for_both_architectures(self):
        for arch in ("aarch64", "x86_64"):
            plan = package.resolve("auto", f"{arch}-apple-darwin")
            with patch.object(package.platform, "system", return_value="Darwin"):
                command = package.cargo("zigbuild", plan)
            self.assertEqual(command[:2], ["cargo", "build"])
            self.assertEqual(command[command.index("--target") + 1], plan["target"])
            with patch.object(package.platform, "system", return_value="Linux"):
                self.assertEqual(package.cargo("zigbuild", plan)[:2], ["cargo", "zigbuild"])
        plan = package.resolve("auto", "x86_64-unknown-linux-gnu")
        with patch.object(package.platform, "system", return_value="Darwin"):
            command = package.cargo("zigbuild", plan)
        self.assertEqual(command[:2], ["cargo", "zigbuild"])
        self.assertIn("x86_64-unknown-linux-gnu.2.28", command)

    def test_build_script_rejects_wrong_features_and_target(self):
        plan = package.resolve("auto", "x86_64-unknown-linux-gnu")
        with tempfile.TemporaryDirectory() as temporary:
            env = {
                **package.build_env(plan),
                "TARGET": plan["target"],
                "HOST": package.rust_host(),
                "OUT_DIR": temporary,
            }
            env.pop("CARGO_FEATURE_CUDA", None)
            result = subprocess.run([package.planner()], env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"backend/features mismatch", result.stderr)
            env["CARGO_FEATURE_CUDA"] = "1"
            env["TARGET"] = "aarch64-unknown-linux-gnu"
            result = subprocess.run([package.planner()], env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"differs from Cargo TARGET", result.stderr)
            env["TARGET"] = plan["target"]
            env["CARGO_FEATURE_TEST_BACKENDS"] = "1"
            result = subprocess.run([package.planner()], env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"backend/features mismatch", result.stderr)

    def test_backend_cfgs_follow_target_and_feature_for_both_build_scripts(self):
        with tempfile.TemporaryDirectory() as temporary:
            cuda = str(Path(temporary) / "cuda-build")
            package.run(["rustc", "--edition", "2024", "crates/backend/cuda/build.rs", "-o", cuda])
            targets = [
                ("x86_64-unknown-linux-gnu", "linux"),
                ("aarch64-unknown-linux-gnu", "linux"),
                ("x86_64-apple-darwin", "macos"),
                ("aarch64-apple-darwin", "macos"),
                ("x86_64-pc-windows-msvc", "windows"),
            ]
            for binary in [str(package.planner()), cuda]:
                for target, target_os in targets:
                    # The packaging contract still rejects Windows production packages.
                    if binary != cuda and target_os == "windows":
                        continue
                    for feature in [False, True]:
                        with self.subTest(binary=binary, target=target, feature=feature):
                            env = {
                                **os.environ,
                                "TARGET": target,
                                "CARGO_CFG_TARGET_OS": target_os,
                                "OUT_DIR": temporary,
                            }
                            for key in [
                                "INFER_PACKAGE_BUILD",
                                "INFER_PACKAGE_TARGET",
                                "CARGO_FEATURE_CUDA",
                            ]:
                                env.pop(key, None)
                            if feature:
                                env["CARGO_FEATURE_CUDA"] = "1"
                            result = subprocess.run(
                                [binary], env=env, capture_output=True, text=True, check=True
                            )
                            flags = set(result.stdout.splitlines())
                            expected = set()
                            if target_os == "linux" and feature:
                                expected = {"cuda_backend", "native_backend"}
                            elif target_os == "macos":
                                expected = {"metal_backend", "native_backend"}
                            emitted = {
                                line.removeprefix("cargo:rustc-cfg=")
                                for line in flags
                                if line.startswith("cargo:rustc-cfg=")
                            }
                            self.assertEqual(emitted, expected)
                            for name in ["cuda_backend", "metal_backend", "native_backend"]:
                                self.assertIn(f"cargo:rustc-check-cfg=cfg({name})", flags)

    def archive(self, directory, *, corrupt=False, extra=None, arch="x86_64"):
        stage = directory / "stage"
        (stage / "bin").mkdir(parents=True)
        binary = stage / "bin/infer"
        binary.write_bytes(elf(arch))
        binary.chmod(0o755)
        manifest = {
            **package.resolve("auto", f"{arch}-unknown-linux-gnu"),
            "files": {"bin/infer": package.digest(binary)},
        }
        (stage / "manifest.json").write_text(json.dumps(manifest))
        if corrupt:
            binary.write_bytes(b"corrupted executable")
        archive = directory / "release.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for file in (binary, stage / "manifest.json"):
                bundle.add(file, arcname="infer/" + str(file.relative_to(stage)))
            if extra:
                member = tarfile.TarInfo(extra)
                member.size = 1
                bundle.addfile(member, io.BytesIO(b"x"))
        return archive

    def test_verified_inventory_and_cross_archive_does_not_execute(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root, arch="aarch64")
            directory, manifest = package.inspect_archive(archive, root / "unpacked")
            self.assertTrue((directory / "bin/infer").is_file())
            self.assertEqual(manifest["arch"], "aarch64")
            with (
                patch.object(package, "rust_host", return_value="x86_64-unknown-linux-gnu"),
                patch.object(package, "smoke") as smoke,
            ):
                package.verify(archive)
                smoke.assert_not_called()
                with self.assertRaisesRegex(ValueError, "target host"):
                    package.verify(archive, "model", "golden")

    def test_binary_architecture_must_match_the_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "infer"
            binary.write_bytes(elf("x86_64"))
            plan = package.resolve("auto", "aarch64-unknown-linux-gnu")
            with self.assertRaisesRegex(ValueError, "architecture"):
                package.check_binary(binary, plan)

    def test_macho_architecture_and_missing_cross_sdk(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "infer"
            header = bytearray(32)
            header[:4] = b"\xcf\xfa\xed\xfe"
            header[4:8] = (0x100000C).to_bytes(4, "little")
            binary.write_bytes(header)
            plan = package.resolve("macos-metal", "aarch64-apple-darwin")
            package.check_binary(binary, plan)
            with self.assertRaisesRegex(ValueError, "architecture"):
                package.check_binary(binary, package.resolve("auto", "x86_64-apple-darwin"))
            result = subprocess.run(
                [package.planner(), "--preflight", plan["target"], "x86_64-unknown-linux-gnu"],
                env={"SDKROOT": temporary},
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"SDKROOT has no Metal.framework", result.stderr)

    def test_corruption_traversal_duplicates_and_unlisted_files_fail(self):
        for corrupt, extra in [
            (True, None),
            (False, "../escape"),
            (False, "infer/bin/infer"),
            (False, "infer/extra"),
        ]:
            with (
                self.subTest(corrupt=corrupt, extra=extra),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                archive = self.archive(root, corrupt=corrupt, extra=extra)
                with self.assertRaises(ValueError):
                    package.inspect_archive(archive, root / "unpacked")

    def test_failed_smoke_never_publishes_a_release(self):
        plan = package.resolve("auto", "x86_64-unknown-linux-gnu")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "infer"
            binary.write_bytes(elf())
            binary.chmod(0o755)
            out = root / "out"
            with (
                patch.object(package, "target_validation", return_value="target-checks"),
                patch.object(package, "build", return_value=binary),
                patch.object(package, "output", return_value="host: test-host"),
                patch.object(package, "verify", side_effect=ValueError("smoke failed")),
                self.assertRaisesRegex(ValueError, "smoke failed"),
            ):
                package.package(plan, out)
            self.assertEqual(list(out.iterdir()), [])

    def test_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "link.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                member = tarfile.TarInfo("infer/bin/infer")
                member.type = tarfile.SYMTYPE
                member.linkname = "/bin/sh"
                bundle.addfile(member)
            with self.assertRaises(ValueError):
                package.inspect_archive(archive, root / "unpacked")


if __name__ == "__main__":
    unittest.main()
