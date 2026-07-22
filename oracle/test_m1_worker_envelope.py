#!/usr/bin/env python3
"""Model-free controls for the live M1 worker provenance envelope."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

isolated_identity = types.ModuleType("_hyperion_isolated_identity")
isolated_identity.ORACLE_STARTUP_IDENTITY = {}
sys.modules["_hyperion_isolated_identity"] = isolated_identity

from m1_bench_worker import (
    EXPECTED_ENVIRONMENT,
    canonical_macos_version,
    verified_startup_environment,
)


class M1WorkerEnvelopeTests(unittest.TestCase):
    def test_startup_snapshot_survives_stock_import_mutation(self) -> None:
        with patch.dict(os.environ, EXPECTED_ENVIRONMENT, clear=True):
            startup = verified_startup_environment()
            os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
            self.assertEqual(startup, EXPECTED_ENVIRONMENT)

    def test_ambient_environment_drift_is_rejected(self) -> None:
        ambient = {**EXPECTED_ENVIRONMENT, "MLX_METAL_DEBUG": "1"}
        with patch.dict(os.environ, ambient, clear=True):
            with self.assertRaisesRegex(RuntimeError, "worker environment must be exactly"):
                verified_startup_environment()

    def test_macos_version_is_canonical_three_component_numeric(self) -> None:
        self.assertEqual(canonical_macos_version("26.6"), "26.6.0")
        self.assertEqual(canonical_macos_version("26.6.1"), "26.6.1")
        self.assertEqual(canonical_macos_version("026.006"), "26.6.0")
        for malformed in ("", "26.6.0.1", "26.6beta", "26..6"):
            with self.subTest(malformed=malformed):
                with self.assertRaisesRegex(RuntimeError, "macOS version is not numeric"):
                    canonical_macos_version(malformed)


if __name__ == "__main__":
    unittest.main()
