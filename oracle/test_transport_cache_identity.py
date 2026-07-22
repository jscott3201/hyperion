#!/usr/bin/env python3
"""Model-free negative controls for exact transport-cache traversal."""

from __future__ import annotations

import errno
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import model_identity


class TransportCacheIdentityTests(unittest.TestCase):
    def make_cache(self, root: Path) -> Path:
        huggingface = root / ".cache" / "huggingface"
        download = huggingface / "download"
        download.mkdir(parents=True)
        (download / "metadata.json").write_text("{}\n", encoding="utf-8")
        return huggingface

    def test_empty_directory_changes_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            huggingface = self.make_cache(root)
            before = model_identity.transport_cache_identity(root)
            (huggingface / "unexpected-empty-directory").mkdir()
            after = model_identity.transport_cache_identity(root)
            self.assertNotEqual(before[0], after[0])
            self.assertEqual(before[1], after[1])

    def test_unreadable_directory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            huggingface = self.make_cache(root)
            unreadable = huggingface / "unreadable"
            unreadable.mkdir()
            (unreadable / "hidden.json").write_text("{}\n", encoding="utf-8")
            real_scandir = os.scandir

            def controlled_scandir(path: os.PathLike[str] | str):
                if Path(path) == unreadable:
                    raise PermissionError(
                        errno.EACCES,
                        "synthetic unreadable directory",
                        os.fspath(path),
                    )
                return real_scandir(path)

            with patch.object(
                model_identity.os,
                "scandir",
                side_effect=controlled_scandir,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "transport cache traversal failed closed"
                ):
                    model_identity.transport_cache_identity(root)


if __name__ == "__main__":
    unittest.main()
