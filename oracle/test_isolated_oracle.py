#!/usr/bin/env python3
"""Model-free negative controls for the M1 Python startup tree."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from isolated_oracle import canonical_tree


class CanonicalTreeTests(unittest.TestCase):
    def test_runtime_hash_binds_mutated_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bytecode = root / "module.pyc"
            bytecode.write_bytes(b"first")
            first = canonical_tree(root, allow_symlinks=True)
            bytecode.write_bytes(b"second")
            second = canonical_tree(root, allow_symlinks=True)
            self.assertNotEqual(first, second)

    def test_site_tree_rejects_regular_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "module.pyc").write_bytes(b"bytecode")
            with self.assertRaisesRegex(RuntimeError, "bytecode file"):
                canonical_tree(root, allow_symlinks=False, reject_bytecode=True)

    def test_site_tree_rejects_standalone_bytecode_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "target").write_bytes(b"target")
            (root / "module.pyc").symlink_to("target")
            with self.assertRaisesRegex(RuntimeError, "file symlink"):
                canonical_tree(root, allow_symlinks=False, reject_bytecode=True)

    def test_site_tree_rejects_symlinked_pycache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "target").mkdir()
            (root / "__pycache__").symlink_to("target", target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "directory symlink"):
                canonical_tree(root, allow_symlinks=False, reject_bytecode=True)


if __name__ == "__main__":
    unittest.main()
