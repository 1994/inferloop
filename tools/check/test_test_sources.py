"""In-src test bodies stay out of production policy scans."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "test_sources", Path(__file__).with_name("test_sources.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestSourcesTest(unittest.TestCase):
    def crate(self, directory, files):
        root = Path(directory)
        source = root / "crates" / "group" / "crate-a" / "src"
        source.mkdir(parents=True)
        for name, text in files.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return root, source

    def test_a_cfg_test_mount_is_a_test_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source = self.crate(
                directory,
                {
                    "lib.rs": "#[cfg(test)]\nmod tests;\n",
                    "tests.rs": "#[test]\nfn case() {}\n",
                },
            )
            self.assertIn((source / "tests.rs").resolve(), module.test_source_paths(root))

    def test_a_feature_gated_test_mount_is_a_test_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source = self.crate(
                directory,
                {
                    "lib.rs": '#[cfg(all(test, feature = "cuda"))]\nmod device_tests;\n',
                    "device_tests.rs": "#[test]\nfn case() {}\n",
                },
            )
            self.assertIn((source / "device_tests.rs").resolve(), module.test_source_paths(root))

    def test_a_mount_shared_with_production_is_not_a_test_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source = self.crate(
                directory,
                {
                    "lib.rs": "#[cfg(any(test, cuda_backend))]\nmod speculation;\n",
                    "speculation.rs": "pub fn check() {}\n",
                },
            )
            self.assertNotIn((source / "speculation.rs").resolve(), module.test_source_paths(root))

    def test_nested_path_mounts_of_a_test_body_are_test_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source = self.crate(
                directory,
                {
                    "lib.rs": "#[cfg(test)]\nmod tests;\n",
                    "tests.rs": '#[path = "helper.rs"]\nmod helper;\n',
                    "helper.rs": "fn values() {}\n",
                },
            )
            sources = module.test_source_paths(root)
            self.assertIn((source / "tests.rs").resolve(), sources)
            self.assertIn((source / "helper.rs").resolve(), sources)


if __name__ == "__main__":
    unittest.main()
