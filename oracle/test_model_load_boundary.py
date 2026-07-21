#!/usr/bin/env python3
"""Prove that a changed model cannot reach even a synthetic loader callback."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from model_identity import verified_model_load, verify_model_tree


class ModelLoadBoundaryTests(unittest.TestCase):
    def test_mutation_after_early_check_never_reaches_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "model.safetensors"
            weights.write_bytes(b"reviewed weights")
            payload_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
            manifest = root / "SHA256SUMS"
            manifest.write_text(
                f"{payload_sha256}  ./model.safetensors\n", encoding="utf-8"
            )
            manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
            early = verify_model_tree(root, manifest_sha256)
            weights.write_bytes(b"mutated after early check")
            called = False

            def synthetic_loader(_path: str) -> object:
                nonlocal called
                called = True
                return object()

            with self.assertRaisesRegex(RuntimeError, "model payload hash differs"):
                verified_model_load(
                    root,
                    manifest_sha256,
                    early,
                    synthetic_loader,
                )
            self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
