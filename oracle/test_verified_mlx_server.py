#!/usr/bin/env python3
"""Model-free controls for the server's in-process model-load guard."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path


class FakeModelProvider:
    def __init__(self, _cli_args: argparse.Namespace) -> None:
        self.loaded = False
        self.mutate_during_load = None

    def _load(
        self,
        _model_path: str,
        _adapter_path: str | None = None,
        _draft_model_path: str | None = None,
    ) -> None:
        self.loaded = True
        if self.mutate_during_load is not None:
            self.mutate_during_load()


fake_core = types.ModuleType("mlx.core")
fake_core.metal = types.SimpleNamespace(is_available=lambda: False)
fake_mlx = types.ModuleType("mlx")
fake_mlx.core = fake_core
fake_server = types.ModuleType("mlx_lm.server")
fake_server.ModelProvider = FakeModelProvider
fake_server.run = lambda *_args, **_kwargs: None
fake_mlx_lm = types.ModuleType("mlx_lm")
fake_mlx_lm.server = fake_server
sys.modules.update(
    {
        "mlx": fake_mlx,
        "mlx.core": fake_core,
        "mlx_lm": fake_mlx_lm,
        "mlx_lm.server": fake_server,
    }
)

from model_identity import verify_model_tree  # noqa: E402
from verified_mlx_server import LOAD_RECEIPT_PREFIX, VerifiedModelProvider  # noqa: E402


def make_model(root: Path) -> tuple[Path, str, dict]:
    weights = root / "model.safetensors"
    weights.write_bytes(b"reviewed server weights")
    payload_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest = root / "SHA256SUMS"
    manifest.write_text(f"{payload_sha256}  ./model.safetensors\n", encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return weights, manifest_sha256, verify_model_tree(root, manifest_sha256)


class VerifiedServerLoadTests(unittest.TestCase):
    def provider(self, root: Path, manifest_sha256: str, identity: dict) -> VerifiedModelProvider:
        return VerifiedModelProvider(
            argparse.Namespace(),
            model_path=root,
            manifest_sha256=manifest_sha256,
            payload_tree_sha256=identity["payload_tree_sha256"],
            payload_file_count=identity["payload_file_count"],
        )

    def test_mutation_after_parent_check_never_reaches_stock_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights, manifest_sha256, early = make_model(root)
            provider = self.provider(root, manifest_sha256, early)
            weights.write_bytes(b"mutated before child load")
            with self.assertRaisesRegex(RuntimeError, "model payload hash differs"):
                provider._load(str(root))
            self.assertFalse(provider.loaded)

    def test_unchanged_tree_loads_and_emits_one_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _weights, manifest_sha256, early = make_model(root)
            provider = self.provider(root, manifest_sha256, early)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                provider._load(str(root))
            self.assertTrue(provider.loaded)
            receipts = [
                line
                for line in stderr.getvalue().splitlines()
                if line.startswith(LOAD_RECEIPT_PREFIX)
            ]
            self.assertEqual(len(receipts), 1)

    def test_mutation_inside_stock_provider_is_rejected_after_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights, manifest_sha256, early = make_model(root)
            provider = self.provider(root, manifest_sha256, early)
            provider.mutate_during_load = lambda: weights.write_bytes(
                b"mutated inside stock provider"
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "server model identity changed while the stock loader ran",
                ):
                    provider._load(str(root))
            self.assertTrue(provider.loaded)
            self.assertNotIn(LOAD_RECEIPT_PREFIX, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
