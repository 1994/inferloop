"""The algorithm-neutrality rule: code in the common layers must not name one algorithm."""

import tempfile
import unittest
from pathlib import Path

import policy


class AlgorithmNeutralityTest(unittest.TestCase):
    def build(self, code: str, layers=None) -> Path:
        root = Path(tempfile.mkdtemp())
        for layer in policy.COMMON_LAYERS if layers is None else layers:
            directory = root / layer
            directory.mkdir(parents=True)
            (directory / "lib.rs").write_text(code)
        return root

    def test_code_naming_an_algorithm_is_rejected(self):
        root = self.build("pub fn draft() -> bool { mtp_depth > 0 }\n")
        with self.assertRaises(SystemExit) as raised:
            policy.check_algorithm_neutrality(root, {})
        self.assertIn("must not name a speculation algorithm", str(raised.exception))

    def test_a_comment_may_name_it(self):
        root = self.build("// MTP draft layers declared by the configuration.\npub fn draft() {}\n")
        policy.check_algorithm_neutrality(root, {})

    def test_a_test_module_may_name_it(self):
        root = self.build("pub fn draft() {}\n#[cfg(test)]\nmod tests {\n    fn mtp() {}\n}\n")
        policy.check_algorithm_neutrality(root, {})

    def test_a_baselined_symbol_passes(self):
        root = self.build("pub mtp_layers: usize,\n", layers=[policy.COMMON_LAYERS[0]])
        policy.check_algorithm_neutrality(root, {"crates/foundation/ir/src/lib.rs": {"mtp_layers"}})

    def test_a_stale_entry_fails_once_the_migration_removes_it(self):
        root = self.build("pub fn draft() {}\n")
        with self.assertRaises(SystemExit) as raised:
            policy.check_algorithm_neutrality(
                root, {"crates/foundation/ir/src/lib.rs": {"mtp_layers"}}
            )
        self.assertIn("Drop the entries", str(raised.exception))

    def test_the_tree_passes(self):
        policy.check_algorithm_neutrality()


class ProductionUnwrapTest(unittest.TestCase):
    def test_a_production_unwrap_is_rejected(self):
        root = Path(tempfile.mkdtemp())
        source = root / "crates/example/src/lib.rs"
        source.parent.mkdir(parents=True)
        source.write_text("pub fn read() -> usize { value().unwrap() }\n")
        with self.assertRaises(SystemExit) as raised:
            policy.check_production_unwraps(root)
        self.assertIn("must propagate this failure", str(raised.exception))

    def test_a_test_source_may_unwrap(self):
        root = Path(tempfile.mkdtemp())
        source = root / "crates/example/tests/unit/case.rs"
        source.parent.mkdir(parents=True)
        source.write_text("fn case() { value().unwrap(); }\n")
        policy.check_production_unwraps(root)

    def test_a_cfg_test_block_in_a_production_file_may_unwrap(self):
        root = Path(tempfile.mkdtemp())
        source = root / "crates/example/src/lib.rs"
        source.parent.mkdir(parents=True)
        body = "pub fn read() -> usize { 1 }\n#[cfg(test)]\nmod tests {\n"
        body += "    fn case() { value().unwrap(); }\n}\n"
        source.write_text(body)
        policy.check_production_unwraps(root)

    def test_the_tree_has_no_production_unwrap(self):
        policy.check_production_unwraps()


class EmbeddedPathTest(unittest.TestCase):
    def test_a_three_segment_call_is_reported(self):
        found = list(policy.embedded_paths("fn f() {\n    super::batch::run();\n}\n"))
        self.assertEqual(found, [(2, "super::batch::run")])

    def test_imports_strings_comments_and_macros_are_not_code(self):
        text = (
            "use super::batch::run;\n"
            "// super::batch::run\n"
            'const A: &str = "super::batch::run \\\n continued";\n'
            "macro_rules! m { () => { $crate::a::b::c() } }\n"
            "fn f() { batch::run() }\n"
        )
        self.assertEqual(list(policy.embedded_paths(text)), [])

    def test_the_tree_has_no_embedded_path(self):
        policy.check_embedded_paths()


if __name__ == "__main__":
    unittest.main()
