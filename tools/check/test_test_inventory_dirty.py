"""The corpus record describes what is committed, not what happens to be on disk."""

import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "test_inventory", Path(__file__).with_name("test-inventory.py")
)
test_inventory = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(test_inventory)


class DirtyPathsTest(unittest.TestCase):
    def test_status_output_is_parsed_into_paths(self):
        porcelain = (
            " M crates/engine/runtime/src/pipeline/mod.rs\n"
            "?? crates/engine/runtime/tests/new.rs\n"
            "MM crates/engine/runtime/src/twice.rs\n"
        )
        self.assertEqual(
            test_inventory.dirty_paths(porcelain),
            {
                "crates/engine/runtime/src/pipeline/mod.rs",
                "crates/engine/runtime/tests/new.rs",
                "crates/engine/runtime/src/twice.rs",
            },
        )

    def test_a_staged_change_is_not_dirty(self):
        # `M ` is what will be committed, which is exactly what the record has to describe.
        self.assertEqual(test_inventory.dirty_paths("M  tools/check/test-inventory.json\n"), set())
        self.assertEqual(test_inventory.dirty_paths("A  crates/x/tests/new.rs\n"), set())

    def test_a_staged_rename_is_not_dirty(self):
        # Renaming is part of what will be committed, so it does not exclude the file.
        porcelain = "R  crates/old/tests/a.rs -> crates/new/tests/b.rs\n"
        self.assertEqual(test_inventory.dirty_paths(porcelain), set())

    def test_an_unstaged_rename_resolves_to_the_new_name(self):
        porcelain = " R crates/old/tests/a.rs -> crates/new/tests/b.rs\n"
        self.assertEqual(test_inventory.dirty_paths(porcelain), {"crates/new/tests/b.rs"})

    def test_a_clean_tree_has_no_paths(self):
        self.assertEqual(test_inventory.dirty_paths(""), set())


class ScanTest(unittest.TestCase):
    def scan(self, dirty):
        crate = Path(test_inventory.REPO_ROOT) / "crates/engine/workloads"
        test_inventory.scan_crate.skipped = set()
        entries, _ignored, _gated, _mounts = test_inventory.scan_crate(crate, dirty)
        return entries, test_inventory.scan_crate.skipped

    def test_an_uncommitted_test_file_is_left_out(self):
        relative = "crates/engine/workloads/src/recurrent_reference_tests.rs"
        included, _ = self.scan(frozenset())
        excluded, skipped = self.scan(frozenset({relative}))
        self.assertIn(relative, skipped)
        self.assertLess(excluded["src_near"], included["src_near"])

    def test_a_clean_scan_skips_nothing(self):
        _, skipped = self.scan(frozenset())
        self.assertEqual(skipped, set())


if __name__ == "__main__":
    unittest.main()
