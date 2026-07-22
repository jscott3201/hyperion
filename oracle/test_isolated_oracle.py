#!/usr/bin/env python3
"""Model-free negative controls for the M1 Python startup tree."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from isolated_oracle import canonical_tree
import isolated_oracle


class CanonicalTreeTests(unittest.TestCase):
    def test_runtime_tree_ignores_bytecode(self) -> None:
        # The runtime root is the shared uv base interpreter prefix; its
        # __pycache__ content accumulates non-deterministically as other tools
        # import the same interpreter, so the runtime tree excludes bytecode.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "module.py"
            source.write_bytes(b"first-source")
            bytecode = root / "module.pyc"
            bytecode.write_bytes(b"first-bytecode")
            base = canonical_tree(root, allow_symlinks=True, ignore_bytecode=True)
            # Mutating bytecode content must not change the runtime tree.
            bytecode.write_bytes(b"second-bytecode")
            self.assertEqual(
                canonical_tree(root, allow_symlinks=True, ignore_bytecode=True),
                base,
            )
            # Adding or removing bytecode must not change the runtime tree.
            bytecode.unlink()
            self.assertEqual(
                canonical_tree(root, allow_symlinks=True, ignore_bytecode=True),
                base,
            )
            # Mutating real source still binds the runtime tree.
            source.write_bytes(b"second-source")
            self.assertNotEqual(
                canonical_tree(root, allow_symlinks=True, ignore_bytecode=True),
                base,
            )

    def test_runtime_tree_prunes_pycache_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "module.py").write_bytes(b"src")
            base = canonical_tree(root, allow_symlinks=True, ignore_bytecode=True)
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "module.cpython-312.pyc").write_bytes(b"compiled")
            self.assertEqual(
                canonical_tree(root, allow_symlinks=True, ignore_bytecode=True),
                base,
            )

    def test_runtime_tree_codesignature_strip_passes_through_non_macho(self) -> None:
        # Signature stripping only applies to Mach-O files; a plain source file
        # must hash identically whether or not strip_codesignature is requested.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "module.py").write_bytes(b"plain source")
            without = canonical_tree(root, allow_symlinks=True, ignore_bytecode=True)
            with_strip = canonical_tree(
                root,
                allow_symlinks=True,
                ignore_bytecode=True,
                strip_codesignature=True,
            )
            self.assertEqual(without, with_strip)

    def test_runtime_tree_normalizes_install_prefix(self) -> None:
        # The uv base prefix is relocated per machine: the install path is
        # HOME-embedded in libpython3.12.dylib and _sysconfigdata. Stripping
        # each tree's own prefix must make two machines with identical code but
        # different install paths hash identically.
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            root_a = Path(a) / "cpython"
            root_b = Path(b) / "cpython"
            root_a.mkdir()
            root_b.mkdir()
            payload = 'EXENAME = "{prefix}/bin/python3"\nDATA = "same"\n'
            (root_a / "sysconfig.py").write_text(payload.format(prefix=root_a))
            (root_b / "sysconfig.py").write_text(payload.format(prefix=root_b))
            normalized_a = canonical_tree(
                root_a,
                allow_symlinks=True,
                ignore_bytecode=True,
                path_prefix=str(root_a),
            )
            normalized_b = canonical_tree(
                root_b,
                allow_symlinks=True,
                ignore_bytecode=True,
                path_prefix=str(root_b),
            )
            self.assertEqual(normalized_a, normalized_b)
            # Without prefix normalization the two machines differ.
            self.assertNotEqual(
                canonical_tree(root_a, allow_symlinks=True, ignore_bytecode=True),
                canonical_tree(root_b, allow_symlinks=True, ignore_bytecode=True),
            )

    def test_validate_expected_relax_skips_runtime_tree(self) -> None:
        # Cross-machine runners relax the runtime-tree pin; a wrong runtime tree
        # must raise in strict mode and be skipped in relaxed mode (the launcher
        # binary and site-packages tree stay pinned in both).
        wrong = {
            "python_executable_sha256": isolated_oracle.EXPECTED_PYTHON_EXECUTABLE_SHA256,
            "python_runtime_tree_sha256": "0" * 64,
            "python_runtime_file_count": 0,
            "site_packages_tree_sha256": isolated_oracle.EXPECTED_SITE_PACKAGES_TREE_SHA256,
            "site_packages_file_count": isolated_oracle.EXPECTED_SITE_PACKAGES_FILE_COUNT,
        }
        isolated_oracle.RELAX_RUNTIME_TREE = False
        with self.assertRaisesRegex(RuntimeError, "python_runtime_tree_sha256"):
            isolated_oracle.validate_expected(dict(wrong))
        isolated_oracle.RELAX_RUNTIME_TREE = True
        try:
            isolated_oracle.validate_expected(dict(wrong))
        finally:
            isolated_oracle.RELAX_RUNTIME_TREE = False

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
